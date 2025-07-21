import pickle
import pandas as pd
import sys

# --- 设置Pandas显示选项以更好地展示前几行 ---
# 不限制显示的列数，以便在单行中看到所有列
pd.set_option('display.max_columns', None)
# 设置一个较大的宽度，防止行被折叠
pd.set_option('display.width', 200)


# 从命令行参数获取文件路径
if len(sys.argv) < 2:
    print("错误: 请提供.pkl文件的路径作为参数。")
    print("用法: python show_pkl_summary.py <path_to_your_file.pkl>")
    sys.exit(1)

file_path = sys.argv[1]

try:
    # 以二进制读取模式打开文件
    with open(file_path, 'rb') as f:
        # 加载pickle文件内容
        data = pickle.load(f)

    print(f"--- 文件摘要: {file_path} ---")
    
    # 检查加载的数据是否为Pandas DataFrame
    if isinstance(data, pd.DataFrame):
        print("\n[+] 文件中的列名:")
        # 使用 tolist() 获得一个清晰的列表输出
        print(data.columns.tolist())
        
        print("\n[+] 文件前三行内容:")
        print(data.head(3))
    else:
        print(f"\n[!] 加载的对象不是Pandas DataFrame，类型为: {type(data)}")
        print("将尝试打印对象的前500个字符:")
        print(str(data)[:500])


except FileNotFoundError:
    print(f"错误: 文件未找到 '{file_path}'")
except Exception as e:
    print(f"加载或处理文件时发生错误: {e}")