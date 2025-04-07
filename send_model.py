import smtplib
import h5py
import os
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

MODEL_PATH = "model/model.h5"
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_email_password"
RECEIVER_EMAIL = "receiver_email@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Check H5 file
try:
    with h5py.File(MODEL_PATH, "r") as f:
        print("Keys in H5 file:", list(f.keys()))
        if "model_weights" in f:
            print("Model weights found ✅")
        if "keras_version" in f.attrs:
            print("Keras version used:", f.attrs["keras_version"])
        if "backend" in f.attrs:
            print("Backend used:", f.attrs["backend"])
except Exception as e:
    print("Error reading H5 file:", str(e))
    exit()

# Send file via email
try:
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = "Model H5 File"

    with open(MODEL_PATH, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(MODEL_PATH)}")
        msg.attach(part)

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
    server.quit()

    print("Email sent successfully! ✅")

except Exception as e:
    print("Error sending email:", str(e))
