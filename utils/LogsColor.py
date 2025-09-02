import loguru
from loguru import logger
import sys

logger_format = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "  # 颜色>时间
                 "{process.name} | "  # 进程名
                 "{thread.name} | "  # 进程名
                 "<cyan>{module}</cyan>.<cyan>{function}</cyan>"  # 模块名.方法名
                 ":<cyan>{line}</cyan> | "  # 行号
                 "<level>{level}</level> | "  # 等级
                 "<level>{message}</level>")  # 日志内容


def get_logger(
        format_type: str | None = None,
        level: str = 'DEBUG',
        debug_file: str = None,
        error_file: str = None,
        info_file: str = None,
):
    """重置loguru配置,定制控制台输出格式,根据传入的日志级别输出到不同的日志文件"""
    logger.remove()
    if format_type is None:
        format_type = logger_format
    logger.add(sys.stdout, level=level, format=format_type, colorize=True, enqueue=True)

    # 按日期保存日志文件
    log_file_format = "logs/loguruLogs/{time:YYYY-MM-DD}.log"  # 按天保存日志
    # 按天切割日志文件，保留最近30天的日志
    logger.add(log_file_format,
               level='DEBUG',
               format=format_type,
               rotation="00:00",  # 每天午夜切割
               retention="30 days",  # 只保留近30天的日志
               enqueue=True)

    if debug_file:
        logger.add(debug_file, level='DEBUG', format=format_type, enqueue=True)
    if error_file:
        logger.add(error_file, level='ERROR', format=format_type.replace('green>', 'yellow>'),
                   enqueue=True)
    if info_file:
        logger.add(info_file, level='INFO', format=format_type, enqueue=True)

    return logger


logging = get_logger()
