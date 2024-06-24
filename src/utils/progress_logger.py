import time


class ProgressLogger:
    last_label = ""
    last_start = -1.0

    @staticmethod
    def start(label: str) -> None:
        ProgressLogger.last_label = label
        ProgressLogger.last_start = time.time()
        print(f"{label}... ⏳", end="\r", flush=True)

    @staticmethod
    def stop() -> None:
        timediff = time.time() - ProgressLogger.last_start
        print(f"{ProgressLogger.last_label} DONE. ✅ Took {timediff:.2f}s.")
