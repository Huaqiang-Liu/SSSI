代码文件：sssi/client_demo2.py

运行示例：

# 无混淆
python client_demo2.py --role host
python client_demo2.py --role guest

# 启用混淆保护（默认情况）
python client_demo2.py --role host --prompt_protect
python client_demo2.py --role guest --prompt_protect

# 启用混淆保护（自定义N值）
python client_demo2.py --role host --prompt_protect --N 8
python client_demo2.py --role guest --prompt_protect --N 8