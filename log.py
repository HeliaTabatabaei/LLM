import datetime
import os

def log_message(message: str, log_file: str = "app_log.txt"):
    """
    یک پیام را به همراه تاریخ و زمان در یک فایل متنی ثبت می‌کند.

    Args:
        message (str): پیامی که می‌خواهید لاگ شود.
        log_file (str, optional): نام فایل لاگ. پیش‌فرض "app_log.txt" است.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    
    try:
        # اگر فایل لاگ وجود نداشت، آن را ایجاد می‌کنیم
        if not os.path.exists(log_file):
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(log_entry)
        else:
            # در غیر این صورت، پیام را به انتهای فایل اضافه می‌کنیم
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
        # print(f"پیام با موفقیت در '{log_file}' ثبت شد.") # در صورت نیاز، پیام موفقیت را نمایش دهید
    except Exception as e:
        print(f"خطا در ثبت لاگ در فایل '{log_file}': {e}")

# --- مثال نحوه استفاده ---
if __name__ == "__main__":
    # برای ثبت خطا
    log_message("خطا: اتصال به پایگاه داده برقرار نشد.", log_file="error_log.txt")
    
    # برای ثبت یک رویداد عادی
    log_message("عملیات ذخیره پیام با موفقیت انجام شد.", log_file="activity_log.txt")
    
    # استفاده از نام فایل پیش‌فرض
    log_message("برنامه شروع به کار کرد.")

    # مثال با شناسه مکالمه
    conversation_id = "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"
    log_message(f"شروع مکالمه با شناسه: {conversation_id}", log_file="chat_log.txt")
