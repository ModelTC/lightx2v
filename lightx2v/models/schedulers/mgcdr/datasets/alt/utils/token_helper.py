# Standard Library
import base64
import datetime
import getpass
import os
import random
import socket
import string
import subprocess

# Import from alt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# 加密密钥（需要32字节的密钥）
ALT_SECRET_KEY = b"0123456789alt0123456789altabcdef"  # 确保密钥长度为32字节


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except Exception:
        return "0.0.0.0"


def get_username():
    def is_git_directory():
        try:
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def get_git_username():
        try:
            result = subprocess.run(
                ["git", "config", "user.name"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def get_system_username():
        # 获取主机的用户名
        return getpass.getuser()

    if is_git_directory():
        git_username = get_git_username()
        if git_username:
            return git_username
        else:
            return get_system_username()
    else:
        return get_system_username()


def generate_random_string(length=16):
    # 包含所有字母和数字的字符集
    characters = string.ascii_letters + string.digits
    # 使用random.choices从字符集中随机选择字符
    random_string = "".join(random.choices(characters, k=length))
    return random_string


# 初始化加密器
def init_cipher(key):
    iv = os.urandom(16)  # 随机生成一个IV（初始化向量）
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    return cipher, iv


default_local_ip, default_username = get_username(), get_local_ip()


# 生成包含时间、用户名和IP信息的token
def generate_token(username=default_username, ip=default_local_ip, SECRET_KEY=ALT_SECRET_KEY):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    random_string = generate_random_string()
    data = f"{current_time}|{username}|{ip}|{random_string}".encode("utf-8")

    # 填充数据
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()

    # 加密数据
    cipher, iv = init_cipher(SECRET_KEY)
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    # 返回base64编码的IV和加密后的数据
    token = base64.urlsafe_b64encode(iv + encrypted_data).decode("utf-8")
    return token


# 解密token
def decrypt_token(token, SECRET_KEY=ALT_SECRET_KEY):
    token_data = base64.urlsafe_b64decode(token)
    iv = token_data[:16]
    encrypted_data = token_data[16:]

    # 解密数据
    cipher = Cipher(algorithms.AES(SECRET_KEY), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

    # 去填充数据
    unpadder = padding.PKCS7(128).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()

    # 返回解密后的字符串信息
    return data.decode("utf-8")


if __name__ == "__main__":
    # token = generate_token()
    # print("Generated Token:", token)

    token = "qXk1kk7OJoMQSn_F-gcJkATna_Pe1QsEwLv5rv_p_7xo44z_irl6j2b8EWmaXLPGLRYfAvTUQDEcy5RJyxPqJg=="

    decrypted_data = decrypt_token(token)
    print("Decrypted Data:", decrypted_data)
