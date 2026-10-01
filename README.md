# 🧠 High on Therapy: 100% Bootstrapped Native Counseling AI Training Engine
### Zero External Weights • Zero Qwen • From-Scratch Pre-training on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Suraj-Mavuleti/highontherapy-ai-training/blob/main/HighOnTherapy_Bootstrapped_From_Scratch.ipynb)

A dedicated, from-scratch pre-training pipeline for a **100% proprietary Decoder-Only Transformer architecture** with completely randomized weights, trained exclusively on clinical psychotherapy and emotional counseling datasets.

---

## 🚀 1-Click Launch

Click the badge above or use this direct link to open the notebook directly in Google Colab:
👉 **[Open High on Therapy Training Notebook in Google Colab](https://colab.research.google.com/github/Suraj-Mavuleti/highontherapy-ai-training/blob/main/HighOnTherapy_Bootstrapped_From_Scratch.ipynb)**

---

## 📈 Scalable Architecture Tiers

Select your tier in Cell 5 (defaults to `TIER = "base"` via dropdown):

| Tier | Parameters | Architecture | Recommended Hardware & Time | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Nano** | **~135M** | 12 layers, 12 heads, hidden 768 | Free Colab T4 GPU (4–8 hours) | Fast proof-of-concept & immediate testing |
| **Base** | **~360M** | 16 layers, 16 heads, hidden 1024 | Colab Pro / A100 / L4 (12–24 hours) | Strong clinical reasoning & nuance |
| **Prime** | **~1.1B** | 22 layers, 32 heads, hidden 2048 | Colab Pro / A100 (Multi-day run) | Master clinical depth across multi-day runs |
| **Ultra** | **~3.2B** | 28 layers, 32 heads, hidden 3072 | Multi-day / Multi-GPU cluster | Full-scale specialized psychological AI |

---

## 🛡️ Multi-Day Endurance Features

- **⏰ 5-Minute Timed Auto-Backup**: Automatically writes model checkpoints and tokenizer directly to Google Drive (`/content/drive/MyDrive/HighOnTherapy_Training/autobackups/autobackup_latest`) every 5 minutes.
- **🚀 Hardware Auto-Upgrader**: Dynamically analyzes GPU VRAM and automatically scales batch size and data loader workers to maximize Tensor Core utilization.
- **🛡️ Auto-Resume**: Automatically detects the latest checkpoint on Google Drive and continues seamlessly from the exact step and epoch after any Colab disconnect.
- **1-Click Push**: Automatically exports and publishes the trained model directly to your Hugging Face account (`DeV-ZEr0/highontherapy-native`).

---

## 👨‍💻 Author & Education

* **Suraj Mavuleti (Dev Zero)**
* 🎓 **Bachelor of Science (BS) in Electronic Systems** — **Indian Institute of Technology, Madras (IIT Madras / IITM)**
* 🌐 **Official Website & Systems Wiki:** [zero.skillissue.gg](https://zero.skillissue.gg)
* 💼 **LinkedIn Profile:** [linkedin.com/in/suraj-mavuleti-b95993320](https://www.linkedin.com/in/suraj-mavuleti-b95993320)
* 🐙 **GitHub:** [@Suraj-Mavuleti](https://github.com/Suraj-Mavuleti)
