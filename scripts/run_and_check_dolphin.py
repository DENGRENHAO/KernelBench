import shutil
import torch
import pydra
import os
import pynvml
import time
from pydra import REQUIRED, Config
from datasets import load_dataset

from src import eval as kernel_eval
from src import utils as kernel_utils
from scripts.generate_baseline_time import measure_program_time
from src.utils import read_file

"""
Run a pair of KernelBench format (problem, solution) to check if solution is correct and compute speedup

You will need two files
1. Reference: PyTorch reference (module Model) implementation with init and input shapes
2. Solution: PyTorch solution (module ModelNew) with inline CUDA Code
Please see examples in src/prompts

The Reference could be either
1. a local file: specify the path to the file
2. a kernelbench problem: specify level and problem id

====================================================
Usage:
1. PyTorch reference is a local file
python3 scripts/run_and_check.py ref_origin=local ref_arch_src_path=src/prompts/model_ex_add.py kernel_src_path=src/prompts/model_new_ex_add.py

2. PyTorch refernece is a kernelbench problem
python3 scripts/run_and_check.py ref_origin=kernelbench level=<level> problem_id=<problem_id> kernel_src_path=<path to model-generated kernel>
====================================================

"""

torch.set_printoptions(precision=4, threshold=10)

class ScriptConfig(Config):
    def __init__(self):

        # Problem and Solution definition
        # Input src origin definition
        self.ref_origin = REQUIRED # either local or kernelbench
        # ref_origin is local, specify local file path
        self.ref_arch_src_path = ""
        # ref_origin is kernelbench, specify level and problem id
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = ""
        self.problem_id = ""
        # Solution src definition
        self.kernel_src_path = ""


        # KernelBench Eval specific
        # number of trials to run for correctness
        self.num_correct_trials = 5
        # number of trials to run for performance
        self.num_perf_trials = 100
        # timeout for each trial
        self.timeout = 300
        # verbose logging
        self.verbose = False
        self.measure_performance = True
        self.build_dir_prefix = "" # if you want to specify a custom build directory
        self.clear_cache = False # TODO

        # Replace with your NVIDIA GPU architecture, e.g. ["Hopper"]
        self.gpu_arch = ["Ada"] 

    def __repr__(self):
        return f"ScriptConfig({self.to_dict()})"

def evaluate_single_sample_src(ref_arch_src: str, kernel_src: str, configs: dict, device: torch.device) -> kernel_eval.KernelExecResult:
    """
    Evaluate a single sample source code against a reference source code
    """

    kernel_hash = str(hash(kernel_src))
    build_dir = os.path.join(configs["build_dir_prefix"], "test_build", kernel_hash)
    
    if configs["clear_cache"]: # fresh kernel build
        print(f"[INFO] Clearing cache for build directory: {build_dir}")
        shutil.rmtree(build_dir, ignore_errors=True)
    
    num_correct_trials = configs["num_correct_trials"]
    num_perf_trials = configs["num_perf_trials"]    
    verbose = configs["verbose"]
    measure_performance = configs["measure_performance"]
    try:
        eval_result = kernel_eval.eval_kernel_against_ref(
        original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            measure_performance=measure_performance,
            verbose=verbose,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            build_dir=build_dir,
            device=device
        )
        return eval_result
    except Exception as e:
        print(f"[WARNING] Last level catch: Some issue evaluating for kernel: {e} ")
        if "CUDA error" in str(e): 
            # NOTE: count this as compilation failure as it is not runnable code
            metadata = {"cuda_error": f"CUDA Error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        }
            eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False, 
                                                metadata=metadata)
            return eval_result
        else:
            metadata = {"other_error": f"error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        }
            eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False, 
                                                metadata=metadata)
            return eval_result


def get_gpu_utilization_info() -> list:
    """
    Get GPU utilization information for all available GPUs.
    Returns list of tuples: (gpu_id, memory_used_mb, memory_total_mb, gpu_utilization_percent)
    """
    try:
        pynvml.nvmlInit()
        gpu_info = []
        
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_mb = mem_info.used // 1024 // 1024
            memory_total_mb = mem_info.total // 1024 // 1024
            
            # Get GPU utilization
            try:
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util_info.gpu
            except:
                gpu_utilization = 0
            
            gpu_info.append((i, memory_used_mb, memory_total_mb, gpu_utilization))
        
        return gpu_info
    except Exception as e:
        print(f"[WARNING] Failed to get GPU utilization info: {e}")
        # Fallback to basic torch GPU detection
        if torch.cuda.is_available():
            return [(i, 0, 1000, 0) for i in range(torch.cuda.device_count())]
        return []

def select_best_gpu() -> torch.device:
    """
    Select the GPU with lowest combined memory usage and computation load.
    Returns torch device object.
    """
    
    gpu_info = get_gpu_utilization_info()
    
    if not gpu_info:
        print("[INFO] No GPU info available, using cuda:0 as fallback")
        return torch.device("cuda:0")
    
    print("[INFO] GPU Status:")
    print("GPU | Memory Used | Memory Total | GPU Util | Score")
    print("-" * 55)
    
    best_gpu = None
    best_score = float('inf')
    
    for gpu_id, mem_used, mem_total, gpu_util in gpu_info:
        # Calculate memory usage percentage
        mem_usage_pct = (mem_used / mem_total) * 100 if mem_total > 0 else 100
        
        # Combined score: weight memory usage more heavily than GPU utilization
        # Lower score is better
        score = (mem_usage_pct * 0.7) + (gpu_util * 0.3)
        
        print(f"{gpu_id:3d} | {mem_used:10d} MB | {mem_total:11d} MB | {gpu_util:7d}% | {score:5.1f}")
        
        if score < best_score:
            best_score = score
            best_gpu = gpu_id
    
    selected_device = torch.device(f"cuda:{best_gpu}")
    print(f"[INFO] Selected GPU {best_gpu} with score {best_score:.1f}")
    
    return selected_device

def wait_for_gpu_availability(target_memory_threshold_mb, 
                            target_util_threshold,
                            max_wait_time) -> torch.device:
    """
    Wait for a GPU to become available with sufficient free memory and low utilization.
    
    Args:
        target_memory_threshold_mb: Minimum free memory required in MB
        target_util_threshold: Maximum GPU utilization percentage acceptable
        max_wait_time: Maximum time to wait in seconds
    
    Returns:
        torch device object
    """
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        gpu_info = get_gpu_utilization_info()
        
        for gpu_id, mem_used, mem_total, gpu_util in gpu_info:
            free_memory = mem_total - mem_used
            
            if free_memory >= target_memory_threshold_mb and gpu_util <= target_util_threshold:
                selected_device = torch.device(f"cuda:{gpu_id}")
                print(f"[INFO] Found available GPU {gpu_id} with {free_memory}MB free memory and {gpu_util}% utilization")
                return selected_device
        
        print(f"[INFO] No suitable GPU found, waiting... ({int(time.time() - start_time)}s elapsed)")
        time.sleep(5)
    
    print(f"[WARNING] Timeout waiting for available GPU, selecting best current option")
    return select_best_gpu()

def run_and_check(config: ScriptConfig):

    print("Running with config", config)

    # Fetch reference and kernel code

    assert config.ref_origin == "local" or config.ref_origin == "kernelbench", "ref_origin must be either local or kernelbench"
    assert config.kernel_src_path != "", "kernel_src_path is required"  
    
    if config.ref_origin == "local":
        assert config.ref_arch_src_path != "", "ref_arch_src_path is required"
        ref_arch_src = read_file(config.ref_arch_src_path)
    elif config.ref_origin == "kernelbench":
        assert config.dataset_name != "", "dataset_name is required"
        assert config.level != "", "level is required"
        assert config.problem_id != "", "problem_id is required"

        # for now use the HuggingFace dataset
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]

        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

        problem_number = int(problem_name.split("_")[0])
        assert problem_number == config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"

        print(f"Fetched problem {config.problem_id} from KernelBench level {config.level}: {problem_name}")


    else:
        raise ValueError("Invalid ref_origin")
    
    kernel_src = read_file(config.kernel_src_path)

    # Start Evaluation
    # device = torch.device("cuda:0") # default device
    device = wait_for_gpu_availability(
        target_memory_threshold_mb=20000,  # Require at least 20GB free memory
        target_util_threshold=50,          # Require GPU utilization < 50%
        max_wait_time=300                  # Wait max 300 seconds
    )
    print(f"[INFO] Using device: {device}")

    kernel_utils.set_gpu_arch(config.gpu_arch)

    print("[INFO] Evaluating kernel against reference code")
    # Evaluate kernel against reference code
    kernel_eval_result = evaluate_single_sample_src(
        ref_arch_src=ref_arch_src,
        kernel_src=kernel_src,
        configs=config.to_dict(),
        device=device
    )
    print(f"[INFO] Kernel evaluation result: {kernel_eval_result}")
    kernel_exec_time = kernel_eval_result.runtime

    if not kernel_eval_result.compiled or not kernel_eval_result.correctness:
        print("[WARNING] Kernel evaluation failed, skipping performance measurement")
        ref_exec_compile_time = -1
        ref_exec_eager_time = -1
    else:
        # Measure baseline time
        print("[INFO] Measuring reference program time")
        # Default using PyTorch Eager here
        ref_time_eager_result = measure_program_time(ref_arch_name="Reference Program", 
                                                    ref_arch_src=ref_arch_src, 
                                                    num_trials=config.num_perf_trials,
                                                    use_torch_compile=False,
                                                    device=device)
        ref_exec_eager_time = ref_time_eager_result.get("mean", None)

        # Measure Torch Compile time
        ref_time_compile_result = measure_program_time(ref_arch_name="Reference Program", 
                                                    ref_arch_src=ref_arch_src, 
                                                    num_trials=config.num_perf_trials,
                                                    use_torch_compile=True,
                                                    torch_compile_backend="inductor",
                                                    torch_compile_options="default",
                                                    device=device)
        ref_exec_compile_time = ref_time_compile_result.get("mean", None)

    print("="*40)
    print(f"[Eval] Kernel eval result: {kernel_eval_result}")
    print("-"*40)
    print(f"[Timing] PyTorch Reference Eager exec time: {ref_exec_eager_time} ms")
    print(f"[Timing] PyTorch Reference torch.compile time: {ref_exec_compile_time} ms")
    print(f"[Timing] Custom Kernel exec time: {kernel_exec_time} ms")
    print("-"*40)   
    
    if kernel_eval_result.correctness:
        print(f"[Speedup] Speedup over eager: {ref_exec_eager_time / kernel_exec_time:.2f}x")
        print(f"[Speedup] Speedup over torch.compile: {ref_exec_compile_time / kernel_exec_time:.2f}x")
    else:
        print("[Speedup] Speedup Not Available as Kernel did not pass correctness")

    print("="*40)
    
    return {
        "kernel_eval_result": kernel_eval_result.model_dump(),
        "ref_exec_eager_time": ref_exec_eager_time,
        "ref_exec_compile_time": ref_exec_compile_time,
        "kernel_exec_time": kernel_exec_time,
        "speedup_over_eager": ref_exec_eager_time / kernel_exec_time if kernel_eval_result.correctness else None,
        "speedup_over_compile": ref_exec_compile_time / kernel_exec_time if kernel_eval_result.correctness else None,
    }