# 测试模块导入
print("开始测试模块导入...")
try:
    import Work0.config
    print("成功导入 Work0.config")
    import Work0.physics
    print("成功导入 Work0.physics")
    print("所有模块导入成功！")
except Exception as e:
    print(f"导入失败: {e}")
