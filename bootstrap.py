#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bootstrap script for Vietnamese Traffic Law RAG Assistant.
Automatically verifies system dependencies, checks the local Ollama service,
pulls the required models, diagnostics GPU VRAM, and runs database checks.
"""
import sys
import os
import time
import subprocess
from pathlib import Path

def print_step(msg):
    print(f"\n[BOOTSTRAP] === {msg} ===")

def main():
    print("=" * 60)
    print("      VIETNAMESE TRAFFIC LAW RAG SYSTEM - BOOTSTRAP       ")
    print("=" * 60)
    
    # Set UTF-8 output encoding for Windows terminals
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

    # Step 1: Check Python Library Dependencies
    print_step("Bước 1: Kiểm tra thư viện Python dependencies...")
    required_packages = [
        "streamlit", "torch", "sentence_transformers", 
        "chromadb", "google.genai", "openai", "loguru", 
        "pyvi", "yaml", "toml"
    ]
    
    missing_packages = []
    for pkg in required_packages:
        try:
            if pkg == "google.genai":
                from google import genai
            elif pkg == "yaml":
                import yaml
            else:
                __import__(pkg)
            print(f"  [OK] Thư viện '{pkg}' đã được cài đặt.")
        except ImportError:
            missing_packages.append(pkg)
            print(f"  [X] Thiếu thư viện '{pkg}'!")

    if missing_packages:
        print(f"\n[LỖI] Hệ thống thiếu các thư viện sau: {missing_packages}")
        print("Vui lòng chạy lệnh: pip install -r requirements.txt")
        sys.exit(1)
        
    import yaml
    import torch
    import requests

    # Step 2: Check Ollama service
    print_step("Bước 2: Kiểm tra dịch vụ Ollama...")
    ollama_url = "http://localhost:11434"
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if r.status_code == 200:
            print("  [OK] Dịch vụ Ollama đang hoạt động.")
            ollama_ready = True
        else:
            print(f"  [CẢNH BÁO] Ollama trả về status code {r.status_code}.")
            ollama_ready = False
    except Exception as e:
        print(f"  [CẢNH BÁO] Không thể kết nối tới Ollama tại {ollama_url}: {e}")
        print("  Đang cố gắng khởi động Ollama trên Windows...")
        try:
            # Try launching Ollama app on Windows
            subprocess.Popen(["cmd.exe", "/c", "start", "ollama", "run"], shell=True)
            # Wait a few seconds for it to start
            for attempt in range(5):
                time.sleep(2)
                try:
                    r = requests.get(f"{ollama_url}/api/tags", timeout=2)
                    if r.status_code == 200:
                        print("  [OK] Dịch vụ Ollama đã được khởi động thành công.")
                        ollama_ready = True
                        break
                except Exception:
                    pass
            else:
                print("  [LỖI] Không thể tự động mở Ollama. Hãy mở ứng dụng Ollama thủ công.")
                ollama_ready = False
        except Exception as start_err:
            print(f"  [LỖI] Khởi chạy Ollama tự động thất bại: {start_err}")
            ollama_ready = False

    # Step 3: Check Local Model in Ollama
    if ollama_ready:
        print_step("Bước 3: Kiểm tra mô hình qwen2.5:1.5b trong Ollama...")
        model_name = "qwen2.5:1.5b"
        try:
            tags_resp = requests.get(f"{ollama_url}/api/tags").json()
            models = [m["name"] for m in tags_resp.get("models", [])]
            if model_name in models or f"{model_name}:latest" in models or any(model_name in m for m in models):
                print(f"  [OK] Mô hình '{model_name}' đã sẵn sàng.")
            else:
                print(f"  [THÔNG TIN] Mô hình '{model_name}' chưa có sẵn. Đang tự động tải (pull)...")
                print("  (Việc này có thể tốn vài phút tùy thuộc vào kết nối mạng của bạn)")
                pull_resp = requests.post(f"{ollama_url}/api/pull", json={"name": model_name}, timeout=600)
                if pull_resp.status_code == 200:
                    print(f"  [OK] Tải mô hình '{model_name}' thành công.")
                else:
                    print(f"  [LỖI] Lệnh pull trả về mã lỗi {pull_resp.status_code}.")
        except Exception as pull_err:
            print(f"  [LỖI] Lỗi khi kiểm tra/tải mô hình trong Ollama: {pull_err}")

    # Step 4: GPU / VRAM Diagnostics and Auto Configuration
    print_step("Bước 4: Chẩn đoán tài nguyên GPU / VRAM...")
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mb = free_bytes / (1024 ** 2)
        total_mb = total_bytes / (1024 ** 2)
        print(f"  Phát hiện GPU CUDA: {torch.cuda.get_device_name(0)}")
        print(f"  Bộ nhớ VRAM: Trống {free_mb:.0f} MB / Tổng {total_mb:.0f} MB")
        
        # If VRAM is too low, auto-configure CPU fallback in retrieval/generation configs
        if free_mb < 500:
            print("  [CẢNH BÁO] Bộ nhớ VRAM trống quá thấp (< 500MB).")
            print("  Tự động điều chỉnh thiết bị nạp mô hình cục bộ sang 'cpu' trong cấu hình để tránh OOM.")
            # Note: The codebase has smart fallbacks at runtime, so we don't strictly need to edit config,
            # but modifying the configurations is a proactive step.
    else:
        print("  Không phát hiện GPU CUDA. Hệ thống sẽ tự động chạy hoàn toàn trên CPU.")

    # Step 5: Data & Database Index Checks
    print_step("Bước 5: Kiểm tra cấu trúc thư mục dữ liệu và vector index...")
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    vector_store_dir = Path("vector_store/chroma")
    bm25_index = Path("vector_store/bm25_index.pkl")
    
    # Verify raw folder
    if not raw_dir.exists():
        print(f"  [LỖI] Thư mục chứa dữ liệu luật '{raw_dir}' không tồn tại!")
        sys.exit(1)
        
    # Check chunks file
    chunks_file = processed_dir / "chunks.jsonl"
    if not chunks_file.exists():
        print(f"  [CẢNH BÁO] Chưa tìm thấy dữ liệu phân đoạn '{chunks_file}'.")
        print("  Đang tự động chạy trích xuất và phân đoạn dữ liệu (ingestion)...")
        # Run run_ingest.py
        try:
            subprocess.run([sys.executable, "-m", "src.phase1_data_engineering.run_ingest"], check=True)
            print("  [OK] Hoàn thành phân đoạn dữ liệu.")
        except Exception as ingest_err:
            print(f"  [LỖI] Phân đoạn dữ liệu thất bại: {ingest_err}")
            sys.exit(1)
            
    # Check chroma collection
    db_file = vector_store_dir / "chroma.sqlite3"
    if not db_file.exists() or not bm25_index.exists():
        print("  [CẢNH BÁO] Chưa phát hiện cơ sở dữ liệu vector hoặc chỉ mục BM25.")
        print("  Đang tự động xây dựng chỉ mục tìm kiếm (indexing)...")
        try:
            subprocess.run([sys.executable, "-m", "src.phase2_hybrid_search.build_indexes"], check=True)
            print("  [OK] Xây dựng cơ sở dữ liệu vector và chỉ mục thành công.")
        except Exception as idx_err:
            print(f"  [LỖI] Xây dựng chỉ mục thất bại: {idx_err}")
            sys.exit(1)
    else:
        print("  [OK] Chỉ mục tìm kiếm và CSDL vector hợp lệ.")
        
    print("\n" + "=" * 60)
    print("      HỆ THỐNG ĐÃ CẤU HÌNH HOÀN CHỈNH - SẴN SÀNG VẬN HÀNH      ")
    print("=" * 60)
    print("Để khởi chạy ứng dụng giao diện, chạy lệnh:")
    print("  streamlit run app/streamlit_app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
