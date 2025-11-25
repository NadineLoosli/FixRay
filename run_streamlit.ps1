cd $(Split-Path -Parent $MyInvocation.MyCommand.Definition)
& .\.venv\Scripts\Activate.ps1
streamlit run src\app\main.py --server.address 127.0.0.1 --server.port 8505
