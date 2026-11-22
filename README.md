# contamguard   
## Quyết định thiết kế
- Tự thêm token `<|endoftext|>` vào đầu câu (add_bos=True) vì Qwen không tự thêm BOS,
  để token thật đầu tiên cũng có log-prob. Đặt add_bos=False để bỏ.