import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import tensorflow as tf
import sys  
import threading
from typing import Optional, Tuple
case_data = {
    'case1': {
        'A_shape': [8, 100, 100],
        'B_shape': [8, 100],
        'data_type': np.float32,
        'left': True
    },
    'case2': {
        'A_shape': [100,100],
        'B_shape': [100],
        'data_type': np.float32,
        'left': False
    },
    'case3': {
        'A_shape': [5, 5],
        'B_shape': [5],
        'data_type': np.float32,
        'left': True
    },
    'case4': {
        'A_shape': [3, 4, 5, 5],
        'B_shape': [3, 4, 5],
        'data_type': np.float32,
        'left': True
    },
    'case5': {
        'A_shape': [100, 100],
        'B_shape': [100],
        'data_type': np.float32,
        'left': True
    },
    'case6': {
        'A_shape': [1, 1, 1],
        'B_shape': [1, 1],
        'data_type': np.float32,
        'left': True
    },
    'case7': {
        'A_shape': [10, 10],
        'B_shape': [10],
        'data_type': np.float32,
        'left': False
    },
}
def run_with_timeout(func, args=(), kwargs={}, timeout=30):
    result = []
    def target():
        try:
            result.append(func(*args, **kwargs))
        except Exception as e:
            result.append(e)
            print("函数执行异常:",e)
    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None
    if isinstance(result[0], Exception):
        raise result[0]
    return result[0]

def verify_result(real_result, golden):
      # 容忍偏差，一般fp16要求绝对误差和相对误差均不超过千分之一
    if golden.dtype == np.float16:
        loss = 1e-3
    else:
        loss = 1e-4
    
    
    result = np.abs(real_result - golden)  # 计算运算结果和预期结果偏差
    # print("result:", result)
    # print("golden:", golden)
    # print("real_result:", real_result)
    deno = np.maximum(np.abs(real_result), np.abs(golden))  # 获取最大值并组成新数组
    result_atol = np.less_equal(result, loss)  # 计算绝对误差
    # result_rtol = np.less_equal(result / np.add(deno, minimum), loss)  # 计算相对误差
    result_rtol = np.less_equal(result / np.add(deno, 1e-10), loss)  # 计算相对误差
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result.size * loss and np.sum(result_atol == False) > real_result.size * loss:  # 误差超出预期时返回打印错误，返回对比失败
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        caseNmae='case'+num
        tensor_input_A = np.random.uniform(-1000, 1000,case_data[caseNmae]['A_shape']).astype(case_data[caseNmae]['data_type'])
        tensor_input_B = np.random.uniform(-1000, 1000,case_data[caseNmae]['B_shape']).astype(case_data[caseNmae]['data_type'])

        golden = torch.linalg.solve_ex(torch.from_numpy(tensor_input_A), torch.from_numpy(tensor_input_B))[0].numpy()
        left =case_data[caseNmae]['left']
        check_error =False
        tensor_input_npu = torch.from_numpy(tensor_input_A).npu()
        tensor_values_npu = torch.from_numpy(tensor_input_B).npu()
        left_npu=torch.tensor(left).npu()
        check_error_npu=torch.tensor(check_error).npu()
        if num == '5':
            tensor_input_npu_tmp = torch.from_numpy(tensor_input_A).npu()
            tensor_input_npu_tmp = torch.from_numpy(tensor_input_B).npu()
            left_npu_tmp=torch.tensor(left).npu()
            check_error_npu_tmp=torch.tensor(check_error).npu()
            output = run_with_timeout(custom_ops_lib.custom_op, args=(tensor_input_npu_tmp, tensor_input_npu_tmp,left_npu_tmp,check_error_npu_tmp), timeout=30)
        # 修改输入
        output = run_with_timeout(custom_ops_lib.custom_op, args=(tensor_input_npu, tensor_values_npu,left_npu,check_error_npu), timeout=30)
        if output is None:
            print(f"{caseNmae} execution timed out!")
        else:
            output = output[0].cpu().numpy()
            if verify_result(output, golden):
                print(f"{caseNmae} verify result pass!")
            else:
                print(f"{caseNmae} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
