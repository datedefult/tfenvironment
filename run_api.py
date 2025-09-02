import uvicorn

if __name__ == "__main__":
    # 在代码中直接启动 Uvicorn 服务器
    uvicorn.run(
        "api:app",  # 模块名:FastAPI 实例
        host="0.0.0.0",  # 监听地址
        port=8001,  # 端口
        reload=True,  # 是否启用热重载（开发环境）
        workers = 1
    )