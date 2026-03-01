import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from safetensors.torch import load_file, save_file
import re
import os
from pathlib import Path
import threading


def convert_key_name(key):
    """
    转换 Kohya → Diffusers 的 key 命名

    输入: lora_unet_layers_0_attention_to_k.lora_down.weight
    输出: diffusion_model.layers.0.attention.to_k.lora_A.weight

    输入: lora_unet_layers_0_feed_forward_w1.lora_up.weight
    输出: diffusion_model.layers.0.feed_forward.w1.lora_B.weight

    输入: lora_unet_layers_0_attention_to_k.alpha
    输出: diffusion_model.layers.0.attention.to_k.alpha
    """

    # 1. 分离 suffix 并转换命名
    if '.lora_down.weight' in key:
        prefix = key.replace('.lora_down.weight', '')
        suffix = '.lora_A.weight'       # ✅ lora_down → lora_A
    elif '.lora_up.weight' in key:
        prefix = key.replace('.lora_up.weight', '')
        suffix = '.lora_B.weight'       # ✅ lora_up → lora_B
    elif '.alpha' in key:
        prefix = key.replace('.alpha', '')
        suffix = '.alpha'
    else:
        return None

    # 2. 去掉 "lora_unet_" 前缀
    if not prefix.startswith('lora_unet_'):
        return None
    prefix = prefix[len('lora_unet_'):]

    # 3. 匹配 layers_{num}_{rest}
    match = re.match(r'layers_(\d+)_(.*)', prefix)
    if not match:
        return None

    layer_num = match.group(1)
    module_part = match.group(2)

    # 4. 转换 module_part
    module_converted = convert_module_path(module_part)
    if module_converted is None:
        return None

    # 5. 组装（包含 diffusion_model 前缀）
    new_key = f"diffusion_model.layers.{layer_num}.{module_converted}{suffix}"
    return new_key


def convert_module_path(module_part):
    """将 Kohya 下划线路径转换为 Diffusers 点号路径"""

    mappings = {
        # attention
        'attention_to_out_0': 'attention.to_out.0',
        'attention_to_q':     'attention.to_q',
        'attention_to_k':     'attention.to_k',
        'attention_to_v':     'attention.to_v',
        # feed_forward
        'feed_forward_w1':    'feed_forward.w1',
        'feed_forward_w2':    'feed_forward.w2',
        'feed_forward_w3':    'feed_forward.w3',
        # adaLN
        'adaLN_modulation_0': 'adaLN_modulation.0',
    }

    result = mappings.get(module_part)
    if result is None:
        print(f"⚠️  未知模块: {module_part}")
    return result


class LoRAConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Z-Image LoRA 转换工具 v1.2")
        self.root.geometry("900x650")

        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.keep_alpha = tk.BooleanVar(value=False)  # 默认不保留

        self.create_widgets()
        self.run_self_test()  # 启动时自动验证

    def run_self_test(self):
        """启动时自动验证转换函数"""
        tests = [
            (
                "lora_unet_layers_0_attention_to_k.lora_down.weight",
                "diffusion_model.layers.0.attention.to_k.lora_A.weight"
            ),
            (
                "lora_unet_layers_5_feed_forward_w1.lora_up.weight",
                "diffusion_model.layers.5.feed_forward.w1.lora_B.weight"
            ),
            (
                "lora_unet_layers_29_attention_to_out_0.lora_down.weight",
                "diffusion_model.layers.29.attention.to_out.0.lora_A.weight"
            ),
            (
                "lora_unet_layers_0_attention_to_q.alpha",
                "diffusion_model.layers.0.attention.to_q.alpha"
            ),
            (
                "lora_unet_layers_10_adaLN_modulation_0.lora_down.weight",
                "diffusion_model.layers.10.adaLN_modulation.0.lora_A.weight"
            ),
        ]

        all_ok = True
        for input_key, expected in tests:
            result = convert_key_name(input_key)
            if result != expected:
                self.log(f"❌ 自检失败!")
                self.log(f"   输入:   {input_key}")
                self.log(f"   期望:   {expected}")
                self.log(f"   实际:   {result}")
                all_ok = False

        if all_ok:
            self.log("✅ 转换函数自检通过（5/5）")
            self.log("   lora_down → lora_A ✓")
            self.log("   lora_up   → lora_B ✓")
            self.log("   diffusion_model 前缀 ✓")
            self.log("   attention.to_k (下划线) ✓")
            self.log("")

    def create_widgets(self):
        # 标题
        title_frame = tk.Frame(self.root, bg="#2196F3", height=60)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)

        tk.Label(title_frame, text="🔄 Z-Image LoRA Key 转换工具 v1.2",
                font=("Arial", 18, "bold"), bg="#2196F3", fg="white").pack(pady=15)

        # 输入
        input_frame = tk.LabelFrame(self.root, text="📁 输入文件",
                                   font=("Arial", 10, "bold"), padx=15, pady=10)
        input_frame.pack(padx=15, pady=10, fill="x")

        tk.Entry(input_frame, textvariable=self.input_file, width=65).pack(side="left", padx=5)
        tk.Button(input_frame, text="浏览", command=self.browse_input, width=6).pack(side="left", padx=2)
        tk.Button(input_frame, text="🔍 分析", command=self.analyze_file,
                 bg="#4CAF50", fg="white", width=8).pack(side="left", padx=2)

        # 输出
        output_frame = tk.LabelFrame(self.root, text="💾 输出目录（可选）",
                                    font=("Arial", 10, "bold"), padx=15, pady=10)
        output_frame.pack(padx=15, pady=5, fill="x")

        tk.Entry(output_frame, textvariable=self.output_dir, width=65).pack(side="left", padx=5)
        tk.Button(output_frame, text="浏览", command=self.browse_output, width=6).pack(side="left")

        # 选项
        options_frame = tk.LabelFrame(self.root, text="⚙️ 选项",
                                     font=("Arial", 10, "bold"), padx=15, pady=10)
        options_frame.pack(padx=15, pady=5, fill="x")

        tk.Checkbutton(options_frame, text="保留 Alpha 值（Diffusers 风格通常不需要）",
                      variable=self.keep_alpha, font=("Arial", 10)).pack(anchor="w")

        # 按钮
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=12)

        tk.Button(button_frame, text="🚀 开始转换", command=self.start_convert,
                 bg="#2196F3", fg="white", font=("Arial", 13, "bold"),
                 width=18, height=2).pack(side="left", padx=10)

        tk.Button(button_frame, text="🗑️ 清空日志", command=self.clear_log,
                 bg="#607D8B", fg="white", font=("Arial", 13, "bold"),
                 width=18, height=2).pack(side="left", padx=10)

        # 进度条
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(padx=15, pady=5, fill="x")

        # 日志
        log_frame = tk.LabelFrame(self.root, text="📋 日志",
                                 font=("Arial", 10, "bold"), padx=10, pady=10)
        log_frame.pack(padx=15, pady=5, fill="both", expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=12,
                                                  wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)

    def browse_input(self):
        f = filedialog.askopenfilename(
            title="选择 LoRA 文件",
            filetypes=[("Safetensors", "*.safetensors"), ("所有", "*.*")]
        )
        if f:
            self.input_file.set(f)
            self.log(f"✅ 已选择: {f}")

    def browse_output(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.output_dir.set(d)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)

    def analyze_file(self):
        path = self.input_file.get()
        if not path:
            messagebox.showwarning("提示", "请先选择文件")
            return

        try:
            self.log("\n" + "="*70)
            sd = load_file(path)
            keys = list(sd.keys())

            # 判断风格
            if any('lora_unet_' in k for k in keys[:10]):
                style = "Kohya ⚠️ 需要转换"
            elif any('diffusion_model.' in k for k in keys[:10]):
                style = "Diffusers ✅"
            else:
                style = "未知 ❓"

            # 统计
            modules = set()
            for k in keys:
                if 'attention' in k: modules.add('attn')
                if 'feed_forward' in k: modules.add('ff')
                if 'adaLN' in k: modules.add('adaLN')

            ranks = set()
            for k, v in sd.items():
                if ('lora_down' in k or 'lora_A' in k) and len(v.shape) >= 2:
                    ranks.add(v.shape[0])

            dtypes = set(str(v.dtype) for v in sd.values())
            alpha_count = sum(1 for k in keys if '.alpha' in k)

            self.log(f"📄 {Path(path).name}")
            self.log(f"🔢 Keys: {len(keys)}  |  Alpha: {alpha_count}")
            self.log(f"💾 dtype: {', '.join(dtypes)}")
            self.log(f"🏷️  风格: {style}")
            self.log(f"🧩 模块: {', '.join(sorted(modules))}")
            self.log(f"📐 Rank: {', '.join(str(r) for r in sorted(ranks))}")

            self.log("\n前5个 keys:")
            for k in keys[:5]:
                self.log(f"  {k}")
            self.log("="*70)

        except Exception as e:
            self.log(f"❌ {e}")

    def start_convert(self):
        if not self.input_file.get():
            messagebox.showwarning("提示", "请先选择文件")
            return
        t = threading.Thread(target=self.convert_file)
        t.daemon = True
        t.start()

    def convert_file(self):
        input_path = self.input_file.get()
        output_dir = self.output_dir.get()
        keep_alpha = self.keep_alpha.get()

        try:
            self.progress.start(10)
            self.log("\n" + "="*70)
            self.log("🚀 开始转换...")

            sd = load_file(input_path)
            new_sd = {}
            skipped = []
            converted = 0
            alpha_kept = 0
            alpha_total = sum(1 for k in sd if k.endswith('.alpha'))

            for key, tensor in sd.items():
                if key.endswith('.alpha'):
                    if keep_alpha:
                        nk = convert_key_name(key)
                        if nk:
                            new_sd[nk] = tensor
                            alpha_kept += 1
                        else:
                            skipped.append(key)
                    continue

                nk = convert_key_name(key)
                if nk:
                    new_sd[nk] = tensor
                    converted += 1
                else:
                    skipped.append(key)

            # 保存
            name = Path(input_path).stem
            out_file = f"{name}_converted.safetensors"
            out_path = os.path.join(output_dir if output_dir else Path(input_path).parent, out_file)

            save_file(new_sd, out_path)
            self.progress.stop()

            self.log("="*70)
            self.log("✅ 转换成功！")
            self.log(f"  权重: {converted}")
            self.log(f"  Alpha: {alpha_kept}/{alpha_total}")
            self.log(f"  跳过: {len(skipped)}")
            self.log(f"  大小: {os.path.getsize(out_path)/1024/1024:.2f} MB")
            self.log(f"\n📁 {out_path}")

            # 验证
            self.log("\n🔍 验证转换结果（前5个 key）:")
            for k in sorted(new_sd.keys())[:5]:
                self.log(f"  ✅ {k}")

            if skipped:
                self.log(f"\n⚠️  跳过:")
                for k in skipped[:5]:
                    self.log(f"  • {k}")

            self.log("="*70)
            messagebox.showinfo("完成", f"转换完成！\n\n{out_path}")

        except Exception as e:
            self.progress.stop()
            self.log(f"❌ {e}")
            messagebox.showerror("错误", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = LoRAConverterApp(root)
    root.mainloop()