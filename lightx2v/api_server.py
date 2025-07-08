import argparse
import sys
import signal
import atexit
from pathlib import Path
from loguru import logger
import uvicorn

from lightx2v.server.api import ApiServer
from lightx2v.server.service import DistributedInferenceService
from lightx2v.server.utils import ProcessManager


def main():
    ProcessManager.register_signal_handler()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cls", type=str, required=True, choices=["wan2.1", "hunyuan", "wan2.1_causvid", "wan2.1_skyreels_v2_df"], default="hunyuan")
    parser.add_argument("--task", type=str, choices=["t2v", "i2v"], default="t2v")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_inference", action="store_true", help="是否在启动API服务器前启动分布式推理服务")
    parser.add_argument("--nproc_per_node", type=int, default=4, help="分布式推理时每个节点的进程数")

    args = parser.parse_args()
    logger.info(f"args: {args}")

    # 初始化服务
    cache_dir = Path(__file__).parent.parent / ".cache"
    inference_service = DistributedInferenceService()

    # 创建API服务器
    api_server = ApiServer()
    api_server.initialize_services(cache_dir, inference_service)

    # 启动分布式推理服务
    if args.start_inference:
        logger.info("正在启动分布式推理服务...")
        success = inference_service.start_distributed_inference(args)
        if not success:
            logger.error("分布式推理服务启动失败，退出程序")
            sys.exit(1)

        # 注册清理函数
        atexit.register(inference_service.stop_distributed_inference)

        # 注册信号处理器
        def signal_handler(signum, frame):
            logger.info(f"接收到信号 {signum}，正在优雅关闭...")
            try:
                inference_service.stop_distributed_inference()
            except Exception as e:
                logger.error(f"关闭分布式推理服务时发生错误: {str(e)}")
            finally:
                sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    # 启动FastAPI服务器
    try:
        logger.info(f"正在启动FastAPI服务器，端口: {args.port}")
        uvicorn.run(api_server.get_app(), host="0.0.0.0", port=args.port, reload=False, workers=1)
    except KeyboardInterrupt:
        logger.info("接收到KeyboardInterrupt，正在关闭服务...")
    except Exception as e:
        logger.error(f"FastAPI服务器运行时发生错误: {str(e)}")
    finally:
        if args.start_inference:
            inference_service.stop_distributed_inference()


if __name__ == "__main__":
    main()
