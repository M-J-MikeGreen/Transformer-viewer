import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk, VERTICAL, HORIZONTAL, BOTH, END, Y, X
import json
import os
import numpy as np
from safetensors import safe_open
import re
from collections import defaultdict, OrderedDict

class ModelHierarchyViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Safetensors模型层次查看器")
        self.root.geometry("1200x800")
        
        # 检查torch支持
        self.torch_available = self.check_torch_availability()
        
        # 数据存储
        self.current_file = None
        self.current_file_info = None
        self.tensor_data_cache = {}  # 缓存张量数据，避免重复加载
        
        # 创建UI
        self.create_widgets()
        
        # 初始提示
        self.show_initial_prompt()
    
    def check_torch_availability(self):
        """检查torch是否可用"""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def create_widgets(self):
        """创建所有UI组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=BOTH, expand=True)
        
        # 顶部工具栏
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=X, pady=(0, 5))
        
        # 文件操作按钮
        btn_frame = ttk.Frame(toolbar)
        btn_frame.pack(side=tk.LEFT)
        
        open_btn = ttk.Button(btn_frame, text="📂 打开模型文件", command=self.open_file, width=15)
        open_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        export_btn = ttk.Button(btn_frame, text="📤 导出JSON", command=self.export_to_json, width=12)
        export_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        copy_btn = ttk.Button(btn_frame, text="📋 复制内容", command=self.copy_content, width=12)
        copy_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # 状态显示
        status_frame = ttk.Frame(toolbar)
        status_frame.pack(side=tk.RIGHT)
        
        torch_status = "✅ Torch可用" if self.torch_available else "⚠️ Torch未安装"
        status_label = ttk.Label(status_frame, text=torch_status, 
                                foreground="green" if self.torch_available else "orange")
        status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # 状态变量
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.LEFT, padx=(5, 0), fill=X, expand=True)
        
        # 主分割窗格 - 左右分割
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=BOTH, expand=True)
        
        # ================ 左侧区域：层次结构树 ================
        left_frame = ttk.Frame(self.paned_window, width=300)
        self.paned_window.add(left_frame, weight=1)
        
        # 树状视图框架
        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # 树状视图标签
        ttk.Label(tree_frame, text="📊 模型层次结构", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        # 树状视图
        self.tree = ttk.Treeview(tree_frame, selectmode='browse')
        self.tree.pack(fill=BOTH, expand=True, side=tk.LEFT)
        
        # 滚动条
        tree_scroll = ttk.Scrollbar(tree_frame, orient=VERTICAL, command=self.tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=Y)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        
        # 配置树状列
        self.tree["columns"] = ("type", "shape")
        self.tree.column("#0", width=250, minwidth=150)
        self.tree.column("type", width=80, minwidth=80, anchor=tk.W)
        self.tree.column("shape", width=100, minwidth=100, anchor=tk.W)
        
        self.tree.heading("#0", text="层/组件", anchor=tk.W)
        self.tree.heading("type", text="类型", anchor=tk.W)
        self.tree.heading("shape", text="形状", anchor=tk.W)
        
        # 绑定树状选择事件
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)
        
        # ================ 右侧区域：详细信息 ================
        right_frame = ttk.Frame(self.paned_window, width=800)
        self.paned_window.add(right_frame, weight=3)
        
        # 右侧标签页
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # === 页1：张量详情 ===
        details_frame = ttk.Frame(self.notebook)
        self.notebook.add(details_frame, text="📝 张量详情")
        
        # 搜索框架
        search_frame = ttk.Frame(details_frame)
        search_frame.pack(fill=X, pady=(0, 5))
        
        ttk.Label(search_frame, text="🔍 快速搜索:", font=('Arial', 10)).pack(side=tk.LEFT, padx=(0, 5))
        
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, font=('Consolas', 10))
        search_entry.pack(side=tk.LEFT, fill=X, expand=True, padx=(0, 10))
        search_entry.bind('<KeyRelease>', self.search_tensors)
        
        clear_search_btn = ttk.Button(search_frame, text="× 清除", command=self.clear_search, width=8)
        clear_search_btn.pack(side=tk.RIGHT)
        
        # 详细信息文本框
        text_frame = ttk.Frame(details_frame)
        text_frame.pack(fill=BOTH, expand=True)
        
        self.details_text = scrolledtext.ScrolledText(
            text_frame, 
            wrap=tk.WORD, 
            font=('Consolas', 10),
            bg='#f8f9fa',
            fg='#212529'
        )
        self.details_text.pack(fill=BOTH, expand=True)
        
        # === 页2：完整数据查看器 ===
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📊 完整数据")
        
        # 数据显示框架
        data_display_frame = ttk.Frame(data_frame)
        data_display_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        # 顶部信息
        info_frame = ttk.Frame(data_display_frame)
        info_frame.pack(fill=X, pady=(0, 5))
        
        self.current_tensor_label = ttk.Label(info_frame, text="未选择张量", font=('Arial', 10, 'bold'))
        self.current_tensor_label.pack(side=tk.LEFT)
        
        self.data_range_label = ttk.Label(info_frame, text="", font=('Arial', 9))
        self.data_range_label.pack(side=tk.RIGHT)
        
        # 数据显示区域
        self.data_text = scrolledtext.ScrolledText(
            data_display_frame, 
            wrap=tk.NONE,  # 不自动换行，方便查看数据
            font=('Consolas', 10),
            bg='#f8f9fa',
            fg='#212529'
        )
        self.data_text.pack(fill=BOTH, expand=True, pady=(0, 5))
        
        # 滑块框架
        slider_frame = ttk.Frame(data_display_frame)
        slider_frame.pack(fill=X, pady=(5, 0))
        
        ttk.Label(slider_frame, text="数据位置:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.position_var = tk.IntVar(value=0)
        self.position_slider = ttk.Scale(
            slider_frame, 
            from_=0, 
            to=100, 
            variable=self.position_var,
            orient=tk.HORIZONTAL,
            command=self.update_data_view
        )
        self.position_slider.pack(side=tk.LEFT, fill=X, expand=True, padx=(0, 10))
        
        self.position_entry = ttk.Entry(slider_frame, textvariable=self.position_var, width=8)
        self.position_entry.pack(side=tk.LEFT)
        self.position_entry.bind('<Return>', self.update_data_view)
        
        ttk.Label(slider_frame, text="/").pack(side=tk.LEFT)
        self.max_position_label = ttk.Label(slider_frame, text="0")
        self.max_position_label.pack(side=tk.LEFT)
        
        # 每页显示数量
        ttk.Label(slider_frame, text=" 每页:").pack(side=tk.LEFT, padx=(10, 0))
        self.page_size_var = tk.IntVar(value=50)
        page_size_entry = ttk.Entry(slider_frame, textvariable=self.page_size_var, width=5)
        page_size_entry.pack(side=tk.LEFT)
        page_size_entry.bind('<Return>', self.update_data_view)
        
        # 配置文本标签样式
        self.setup_text_tags()
        
        # 右键菜单
        self.create_context_menus()
    
    def setup_text_tags(self):
        """设置文本标签样式"""
        # 详情文本框
        self.details_text.tag_configure('header', foreground='#1e3a8a', font=('Arial', 11, 'bold'))
        self.details_text.tag_configure('subheader', foreground='#047857', font=('Arial', 10, 'bold'))
        self.details_text.tag_configure('path', foreground='#6b7280', font=('Consolas', 9))
        self.details_text.tag_configure('size', foreground='#8b5cf6', font=('Arial', 10, 'bold'))
        self.details_text.tag_configure('tensor_name', foreground='#047857', font=('Consolas', 10, 'bold'))
        self.details_text.tag_configure('dtype', foreground='#dc2626', font=('Consolas', 10))
        self.details_text.tag_configure('shape', foreground='#0ea5e9', font=('Consolas', 10))
        self.details_text.tag_configure('sample', foreground='#84cc16', font=('Consolas', 10))
        self.details_text.tag_configure('error', foreground='#ef4444', font=('Arial', 10, 'bold'))
        self.details_text.tag_configure('warning', foreground='#f59e0b', font=('Arial', 10, 'bold'))
        self.details_text.tag_configure('success', foreground='#10b981', font=('Arial', 10, 'bold'))
        self.details_text.tag_configure('metadata_key', foreground='#8b5cf6', font=('Arial', 10, 'bold'))
        self.details_text.tag_configure('layer_name', foreground='#9333ea', font=('Consolas', 10, 'bold'))
        self.details_text.tag_configure('component', foreground='#dc2626', font=('Consolas', 10))
        self.details_text.tag_configure('value', foreground='#047857', font=('Consolas', 10))
        
        # 数据文本框
        self.data_text.tag_configure('data_header', foreground='#1e3a8a', font=('Consolas', 10, 'bold'))
        self.data_text.tag_configure('data_index', foreground='#6b7280', font=('Consolas', 9))
        self.data_text.tag_configure('data_value', foreground='#047857', font=('Consolas', 10))
        self.data_text.tag_configure('data_highlight', background='#fef3c7', foreground='#92400e')
        self.data_text.tag_configure('data_error', foreground='#ef4444', font=('Consolas', 10, 'bold'))
    
    def create_context_menus(self):
        """创建右键菜单"""
        # 树状视图右键菜单
        self.tree_menu = tk.Menu(self.tree, tearoff=0)
        self.tree_menu.add_command(label="展开全部", command=self.expand_all_tree)
        self.tree_menu.add_command(label="折叠全部", command=self.collapse_all_tree)
        self.tree_menu.add_separator()
        self.tree_menu.add_command(label="复制层名称", command=self.copy_tree_item_name)
        
        self.tree.bind('<Button-3>', self.show_tree_menu)
        
        # 详情文本框右键菜单
        self.details_menu = tk.Menu(self.details_text, tearoff=0)
        self.details_menu.add_command(label="复制选中内容", command=lambda: self.details_text.event_generate('<<Copy>>'))
        self.details_menu.add_command(label="全选", command=lambda: self.details_text.event_generate('<<SelectAll>>'))
        
        self.details_text.bind('<Button-3>', self.show_details_menu)
        
        # 数据文本框右键菜单
        self.data_menu = tk.Menu(self.data_text, tearoff=0)
        self.data_menu.add_command(label="复制选中内容", command=lambda: self.data_text.event_generate('<<Copy>>'))
        self.data_menu.add_command(label="全选", command=lambda: self.data_text.event_generate('<<SelectAll>>'))
        self.data_menu.add_command(label="复制完整数据", command=self.copy_full_data)
        
        self.data_text.bind('<Button-3>', self.show_data_menu)
    
    def show_tree_menu(self, event):
        """显示树状视图右键菜单"""
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.tree_menu.post(event.x_root, event.y_root)
    
    def show_details_menu(self, event):
        """显示详情文本框右键菜单"""
        self.details_menu.post(event.x_root, event.y_root)
    
    def show_data_menu(self, event):
        """显示数据文本框右键菜单"""
        self.data_menu.post(event.x_root, event.y_root)
    
    def copy_tree_item_name(self):
        """复制树状项名称"""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            item_text = self.tree.item(item, 'text')
            self.root.clipboard_clear()
            self.root.clipboard_append(item_text)
            self.status_var.set(f"✅ 复制: {item_text}")
    
    def expand_all_tree(self):
        """展开所有树节点"""
        def expand_children(item):
            self.tree.item(item, open=True)
            for child in self.tree.get_children(item):
                expand_children(child)
        
        for item in self.tree.get_children():
            expand_children(item)
        self.status_var.set("✅ 展开所有节点")
    
    def collapse_all_tree(self):
        """折叠所有树节点"""
        def collapse_children(item):
            self.tree.item(item, open=False)
            for child in self.tree.get_children(item):
                collapse_children(child)
        
        for item in self.tree.get_children():
            collapse_children(item)
        self.status_var.set("✅ 折叠所有节点")
    
    def copy_full_data(self):
        """复制完整数据"""
        content = self.data_text.get(1.0, END)
        if content.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.status_var.set("✅ 完整数据已复制到剪贴板")
    
    def show_initial_prompt(self):
        """显示初始提示信息"""
        self.details_text.delete(1.0, END)
        self.details_text.insert(END, "🚀 欢迎使用Safetensors模型层次查看器\n", 'header')
        self.details_text.insert(END, "=" * 80 + "\n\n")
        
        self.details_text.insert(END, "💡 使用指南:\n\n")
        self.details_text.insert(END, "1. 点击左侧 📂 打开模型文件 按钮加载Safetensors文件\n")
        self.details_text.insert(END, "2. 左侧树状结构显示模型层次：\n")
        self.details_text.insert(END, "   • 📁 embed_tokens: 词嵌入层\n")
        self.details_text.insert(END, "   • 📁 layers.X: Transformer层 (X从0开始)\n")
        self.details_text.insert(END, "   • 📁 self_attn: 自注意力机制 (Q,K,V,O投影)\n")
        self.details_text.insert(END, "   • 📁 mlp: 前馈网络 (gate, up, down投影)\n")
        self.details_text.insert(END, "   • 📁 norm: LayerNorm层\n")
        self.details_text.insert(END, "3. 点击任意层/组件查看详细信息\n")
        self.details_text.insert(END, "4. 切换到 📊 完整数据 标签页查看完整浮点数向量\n")
        self.details_text.insert(END, "5. 使用滑块或输入位置来滚动查看长向量\n\n")
        
        self.details_text.insert(END, "✅ 当前状态: ")
        self.details_text.insert(END, f"Torch支持: {'✅ 可用' if self.torch_available else '⚠️ 未安装'}\n", 
                                'success' if self.torch_available else 'warning')
        self.details_text.insert(END, f"  bfloat16处理: {'✅ 原生支持' if self.torch_available else '⚠️ 转换为float32'}\n",
                                'success' if self.torch_available else 'warning')
        
        self.details_text.see(1.0)
    
    def parse_model_structure(self, tensor_names):
        """解析模型层次结构 - 改进版本"""
        structure = defaultdict(dict)
        
        # 首先收集所有层索引
        layer_indices = set()
        for name in tensor_names:
            if 'layers' in name:
                # 使用正则表达式提取层索引
                match = re.search(r'layers\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                    layer_indices.add(layer_idx)
        
        # 按数字顺序排序层
        sorted_layer_indices = sorted(layer_indices)
        print(f"找到的层索引: {sorted_layer_indices}")  # 调试信息
        
        # 处理每个张量
        for name in tensor_names:
            if 'error' in name:  # 跳过错误的张量
                continue
            
            # 1. 词嵌入层
            if name.startswith('embed_tokens'):
                structure['embed_tokens'][name] = name
            
            # 2. 最终归一化
            elif name.startswith('norm'):
                structure['final_norm'][name] = name
            
            # 3. 语言模型头
            elif name.startswith('lm_head'):
                structure['lm_head'][name] = name
            
            # 4. Transformer层
            elif 'layers' in name:
                # 提取层索引
                match = re.search(r'layers\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                    layer_key = f"Layer {layer_idx}"
                    
                    if layer_key not in structure:
                        structure[layer_key] = {}
                    
                    # 提取组件类型
                    if 'self_attn' in name:
                        if 'q_proj' in name:
                            structure[layer_key]['self_attn.q_proj'] = name
                        elif 'k_proj' in name:
                            structure[layer_key]['self_attn.k_proj'] = name
                        elif 'v_proj' in name:
                            structure[layer_key]['self_attn.v_proj'] = name
                        elif 'o_proj' in name:
                            structure[layer_key]['self_attn.o_proj'] = name
                        elif 'q_norm' in name:
                            structure[layer_key]['self_attn.q_norm'] = name
                        elif 'k_norm' in name:
                            structure[layer_key]['self_attn.k_norm'] = name
                    elif 'mlp' in name:
                        if 'gate_proj' in name:
                            structure[layer_key]['mlp.gate_proj'] = name
                        elif 'up_proj' in name:
                            structure[layer_key]['mlp.up_proj'] = name
                        elif 'down_proj' in name:
                            structure[layer_key]['mlp.down_proj'] = name
                    elif 'input_layernorm' in name:
                        structure[layer_key]['input_layernorm'] = name
                    elif 'post_attention_layernorm' in name:
                        structure[layer_key]['post_attention_layernorm'] = name
            
            # 5. 其他组件
            else:
                if 'other' not in structure:
                    structure['other'] = {}
                structure['other'][name] = name
        
        # 调试信息
        print("解析后的结构:")
        for key, value in structure.items():
            print(f"{key}: {len(value)} 个组件")
        
        return structure
    
    def build_tree_structure(self, structure):
        """构建树状结构"""
        self.tree.delete(*self.tree.get_children())
        
        # 添加根节点
        root_node = self.tree.insert("", "end", "root", text="模型结构", open=True)
        
        # 按顺序添加主要组件
        main_components = [
            ('embed_tokens', '🔤 词嵌入层 (embed_tokens)'),
            ('layers', '🧱 Transformer层'),
            ('final_norm', '🎯 最终归一化 (norm)'),
            ('lm_head', '🎯 语言模型头 (lm_head)'),
            ('other', '📦 其他组件')
        ]
        
        # 1. 词嵌入层
        if 'embed_tokens' in structure:
            embed_node = self.tree.insert(root_node, "end", "embed_tokens", text="🔤 词嵌入层 (embed_tokens)", open=True)
            for tensor_name in structure['embed_tokens'].values():
                self.tree.insert(embed_node, "end", tensor_name, 
                               text=tensor_name,
                               values=('weight', ''))
        
        # 2. Transformer层 - 按数字顺序
        layer_nodes = {}
        for key in sorted(structure.keys()):
            if key.startswith('Layer '):
                layer_nodes[key] = structure[key]
        
        if layer_nodes:
            layers_node = self.tree.insert(root_node, "end", "layers", text="🧱 Transformer层", open=False)
            
            # 按层索引排序
            sorted_layers = sorted(layer_nodes.keys(), key=lambda x: int(x.split()[1]))
            
            for layer_key in sorted_layers:
                layer_info = layer_nodes[layer_key]
                layer_idx = int(layer_key.split()[1])
                layer_node = self.tree.insert(layers_node, "end", layer_key, 
                                             text=f"🧱 Layer {layer_idx}", open=False)
                
                # 按固定顺序添加组件
                component_order = [
                    ('input_layernorm', '📊 输入LayerNorm'),
                    ('self_attn.q_norm', '🟡 Q归一化'),
                    ('self_attn.k_norm', '🔵 K归一化'),
                    ('self_attn.q_proj', '🟡 Q投影'),
                    ('self_attn.k_proj', '🔵 K投影'),
                    ('self_attn.v_proj', '🟢 V投影'),
                    ('self_attn.o_proj', '🔴 O投影'),
                    ('post_attention_layernorm', '📊 注意力后LayerNorm'),
                    ('mlp.gate_proj', '🟠 MLP Gate'),
                    ('mlp.up_proj', '🟠 MLP Up'),
                    ('mlp.down_proj', '⬛ MLP Down')
                ]
                
                added_components = False
                for comp_key, comp_name in component_order:
                    if comp_key in layer_info:
                        tensor_name = layer_info[comp_key]
                        self.tree.insert(layer_node, "end", tensor_name, 
                                       text=comp_name,
                                       values=('weight', ''))
                        added_components = True
                
                # 如果没有按顺序添加的组件，添加其他组件
                if not added_components:
                    for comp_key, tensor_name in layer_info.items():
                        display_name = comp_key.replace('self_attn.', '').replace('mlp.', '')
                        self.tree.insert(layer_node, "end", tensor_name, 
                                       text=f"🔧 {display_name}",
                                       values=('weight', ''))
        
        # 3. 最终归一化
        if 'final_norm' in structure:
            norm_node = self.tree.insert(root_node, "end", "final_norm", text="🎯 最终归一化 (norm)", open=True)
            for tensor_name in structure['final_norm'].values():
                self.tree.insert(norm_node, "end", tensor_name, 
                               text=tensor_name,
                               values=('weight', ''))
        
        # 4. 语言模型头
        if 'lm_head' in structure:
            lm_head_node = self.tree.insert(root_node, "end", "lm_head", text="🎯 语言模型头 (lm_head)", open=True)
            for tensor_name in structure['lm_head'].values():
                self.tree.insert(lm_head_node, "end", tensor_name, 
                               text=tensor_name,
                               values=('weight', ''))
        
        # 5. 其他组件
        other_items = []
        for key, value in structure.items():
            if key not in ['embed_tokens', 'final_norm', 'lm_head'] and not key.startswith('Layer '):
                other_items.append((key, value))
        
        if other_items:
            other_node = self.tree.insert(root_node, "end", "other", text="📦 其他组件", open=False)
            for key, tensor_dict in other_items:
                if isinstance(tensor_dict, dict):
                    sub_node = self.tree.insert(other_node, "end", key, text=f"📁 {key}", open=False)
                    for tensor_name in tensor_dict.values():
                        self.tree.insert(sub_node, "end", tensor_name, 
                                       text=tensor_name,
                                       values=('weight', ''))
                else:
                    self.tree.insert(other_node, "end", key, 
                                   text=f"🔧 {key}",
                                   values=('weight', ''))
        
        # 调试信息
        print(f"树状结构构建完成，节点数: {len(self.tree.get_children())}")
    
    def open_file(self):
        """打开safetensors文件"""
        file_path = filedialog.askopenfilename(
            title="选择Safetensors文件",
            filetypes=[
                ("Safetensors文件", "*.safetensors"),
                ("所有文件", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        self.current_file = file_path
        self.status_var.set(f"⏳ 正在加载: {os.path.basename(file_path)}...")
        self.root.update()
        
        try:
            # 读取文件信息
            file_info = self.read_safetensors_file(file_path)
            self.current_file_info = file_info
            
            # 解析模型结构
            tensor_names = [t['name'] for t in file_info['tensors'] if 'error' not in t]
            print(f"找到 {len(tensor_names)} 个张量")  # 调试信息
            print("前10个张量名称:", tensor_names[:10])  # 调试信息
            
            structure = self.parse_model_structure(tensor_names)
            self.build_tree_structure(structure)
            
            self.status_var.set(f"✅ 加载成功: {len(file_info['tensors'])} 个张量 | {len(structure)} 个组件")
            
            # 显示文件概览
            self.show_file_overview(file_info, file_path)
            
        except Exception as e:
            error_msg = f"❌ 错误: {str(e)}"
            self.details_text.delete(1.0, END)
            self.details_text.insert(END, error_msg + "\n", 'error')
            self.status_var.set("❌ 加载失败")
            messagebox.showerror("错误", str(e))
    
    def read_safetensors_file(self, file_path):
        """读取safetensors文件信息"""
        file_info = {
            'metadata': {},
            'tensors': [],
            'file_size': os.path.getsize(file_path),
            'bfloat16_count': 0
        }
        
        try:
            framework = "pt" if self.torch_available else "numpy"
            
            with safe_open(file_path, framework=framework) as f:
                # 获取元数据
                metadata = f.metadata()
                file_info['metadata'] = metadata if metadata else {}
                print(f"元数据: {metadata}")  # 调试信息
                
                # 获取所有张量
                tensor_names = list(f.keys())
                print(f"文件中的张量名称: {tensor_names[:10]}... (共 {len(tensor_names)} 个)")  # 调试信息
                
                for name in tensor_names:
                    try:
                        if framework == "pt":
                            import torch
                            tensor = f.get_tensor(name)
                            
                            if tensor.dtype == torch.bfloat16:
                                file_info['bfloat16_count'] += 1
                                tensor = tensor.to(torch.float32)
                            
                            # 获取基本形状和大小信息，不加载完整数据
                            tensor_info = {
                                'name': name,
                                'shape': list(tensor.shape),
                                'dtype': str(tensor.dtype).replace("torch.", ""),
                                'size_bytes': tensor.numel() * tensor.element_size(),
                                'has_data': True  # 标记有数据，但不立即加载
                            }
                        else:
                            tensor = f.get_slice(name)
                            dtype_str = str(tensor.dtype)
                            
                            if 'bfloat16' in dtype_str.lower():
                                file_info['bfloat16_count'] += 1
                            
                            # 只获取形状信息，不加载完整数据
                            tensor_info = {
                                'name': name,
                                'shape': tensor.shape,
                                'dtype': dtype_str,
                                'size_bytes': 0,  # 稍后计算
                                'has_data': True
                            }
                        
                        file_info['tensors'].append(tensor_info)
                        
                    except Exception as tensor_error:
                        tensor_info = {
                            'name': name,
                            'error': str(tensor_error),
                            'shape': '未知',
                            'dtype': '未知',
                            'size_bytes': 0,
                            'has_data': False
                        }
                        file_info['tensors'].append(tensor_info)
                        continue
        
        except Exception as e:
            print(f"读取文件时出错: {str(e)}")  # 调试信息
            raise Exception(f"读取文件失败: {str(e)}")
        
        return file_info
    
    def load_tensor_data(self, tensor_name):
        """加载张量的完整数据"""
        if tensor_name in self.tensor_data_cache:
            return self.tensor_data_cache[tensor_name]
        
        if not self.current_file or not self.current_file_info:
            return None
        
        try:
            framework = "pt" if self.torch_available else "numpy"
            
            with safe_open(self.current_file, framework=framework) as f:
                if framework == "pt":
                    import torch
                    tensor = f.get_tensor(tensor_name)
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.to(torch.float32)
                    tensor_array = tensor.cpu().numpy()
                else:
                    tensor = f.get_slice(tensor_name).numpy()
                    if tensor.dtype == np.float16 or tensor.dtype == np.dtype('bfloat16'):
                        tensor = tensor.astype(np.float32)
                
                # 缓存数据
                self.tensor_data_cache[tensor_name] = tensor_array
                return tensor_array
                
        except Exception as e:
            self.status_var.set(f"❌ 加载数据失败: {str(e)}")
            return None
    
    def show_file_overview(self, file_info, file_path):
        """显示文件概览信息"""
        self.details_text.delete(1.0, END)
        
        # 文件基本信息
        self.details_text.insert(END, "📁 文件概览\n", 'header')
        self.details_text.insert(END, "=" * 80 + "\n")
        self.details_text.insert(END, "路径: ", 'subheader')
        self.details_text.insert(END, f"{file_path}\n", 'path')
        self.details_text.insert(END, "大小: ", 'subheader')
        self.details_text.insert(END, f"{file_info['file_size'] / 1024 / 1024:.2f} MB\n", 'size')
        self.details_text.insert(END, "张量总数: ", 'subheader')
        self.details_text.insert(END, f"{len(file_info['tensors'])}\n", 'size')
        
        if file_info['bfloat16_count'] > 0:
            self.details_text.insert(END, "bfloat16张量: ", 'subheader')
            status = "✅ 使用Torch处理" if self.torch_available else "⚠️ 转换为float32显示"
            self.details_text.insert(END, f"{file_info['bfloat16_count']} 个 ({status})\n", 'warning' if not self.torch_available else 'success')
        
        self.details_text.insert(END, "=" * 80 + "\n\n")
        
        # 元数据信息
        self.details_text.insert(END, ".Metadata 信息\n", 'header')
        self.details_text.insert(END, "-" * 40 + "\n")
        
        if file_info['metadata']:
            for key, value in file_info['metadata'].items():
                self.details_text.insert(END, f"{key}: ", 'metadata_key')
                try:
                    json_value = json.loads(value)
                    formatted_value = json.dumps(json_value, indent=2, ensure_ascii=False)
                    self.details_text.insert(END, f"{formatted_value}\n")
                except:
                    self.details_text.insert(END, f"{value}\n")
        else:
            self.details_text.insert(END, "没有元数据\n")
        
        self.details_text.insert(END, "\n" + "=" * 80 + "\n\n")
        
        # 模型结构概览
        self.details_text.insert(END, "🧱 模型结构概览\n", 'header')
        self.details_text.insert(END, "-" * 40 + "\n")
        self.details_text.insert(END, "• 词嵌入层 (embed_tokens)\n")
        self.details_text.insert(END, "• Transformer层 (layers.0 到 layers.N)\n")
        self.details_text.insert(END, "  - 输入LayerNorm (input_layernorm)\n")
        self.details_text.insert(END, "  - 自注意力机制 (self_attn)\n")
        self.details_text.insert(END, "    · Q/K/V/O投影\n")
        self.details_text.insert(END, "    · Q/K归一化\n")
        self.details_text.insert(END, "  - MLP前馈网络 (mlp)\n")
        self.details_text.insert(END, "    · Gate/Up/Down投影\n")
        self.details_text.insert(END, "  - 注意力后LayerNorm (post_attention_layernorm)\n")
        self.details_text.insert(END, "• 最终LayerNorm (norm)\n")
        self.details_text.insert(END, "• 语言模型头 (lm_head, 可选)\n")
        
        self.details_text.insert(END, "\n💡 点击左侧树状结构中的任意组件查看详细信息\n", 'warning')
        
        self.details_text.see(1.0)
    
    def on_tree_select(self, event):
        """树状项选择事件"""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = selection[0]
        item_text = self.tree.item(item, 'text')
        tensor_name = item
        
        # 检查是否是张量节点（不是文件夹）
        # 如果是文件夹节点，不显示张量详情
        parent = self.tree.parent(item)
        if parent == "" or parent == "root" or item in ["embed_tokens", "layers", "final_norm", "lm_head", "other"]:
            # 显示组件信息
            self.show_component_info(item_text)
            return
        
        # 是张量节点，显示张量详情
        self.show_tensor_details(tensor_name)
    
    def show_tensor_details(self, tensor_name):
        """显示张量详细信息"""
        if not self.current_file_info:
            return
        
        # 查找张量信息
        tensor_info = None
        for t in self.current_file_info['tensors']:
            if t['name'] == tensor_name:
                tensor_info = t
                break
        
        if not tensor_info or 'error' in tensor_info:
            self.details_text.delete(1.0, END)
            self.details_text.insert(END, f"❌ 找不到张量: {tensor_name}\n", 'error')
            if tensor_info and 'error' in tensor_info:
                self.details_text.insert(END, f"错误: {tensor_info['error']}\n", 'error')
            return
        
        self.details_text.delete(1.0, END)
        
        # 显示张量详情
        self.details_text.insert(END, f"📊 张量详情: {tensor_name}\n", 'header')
        self.details_text.insert(END, "=" * 80 + "\n\n")
        
        self.details_text.insert(END, "🔤 名称: ", 'subheader')
        self.details_text.insert(END, f"{tensor_name}\n", 'tensor_name')
        
        self.details_text.insert(END, "📏 形状: ", 'subheader')
        self.details_text.insert(END, f"{tensor_info['shape']}\n", 'shape')
        
        self.details_text.insert(END, "🔢 数据类型: ", 'subheader')
        dtype_display = tensor_info['dtype']
        if 'bfloat16' in dtype_display.lower() and not self.torch_available:
            dtype_display += " (已转换为float32)"
        self.details_text.insert(END, f"{dtype_display}\n", 'dtype')
        
        if tensor_info['size_bytes'] > 0:
            self.details_text.insert(END, "💾 大小: ", 'subheader')
            self.details_text.insert(END, f"{tensor_info['size_bytes'] / 1024:.2f} KB\n", 'size')
        
        # 添加操作提示
        self.details_text.insert(END, "\n" + "=" * 80 + "\n")
        self.details_text.insert(END, "💡 操作提示:\n", 'warning')
        self.details_text.insert(END, "• 切换到 '📊 完整数据' 标签页查看完整浮点数向量\n")
        self.details_text.insert(END, "• 使用滑块或输入位置来滚动查看长向量\n")
        self.details_text.insert(END, "• 右键点击数据区域可复制完整数据\n")
        
        # 准备数据查看器
        self.current_tensor_name = tensor_name
        self.current_tensor_info = tensor_info
        self.current_tensor_label.config(text=f"当前张量: {tensor_name}")
        
        # 加载数据预览
        self.load_data_preview(tensor_name)
        
        self.details_text.see(1.0)
    
    def load_data_preview(self, tensor_name):
        """加载数据预览"""
        try:
            tensor_array = self.load_tensor_data(tensor_name)
            if tensor_array is None:
                return
            
            # 获取前100个元素作为预览
            flat_data = tensor_array.flatten()
            preview_count = min(100, len(flat_data))
            preview_data = flat_data[:preview_count]
            
            # 更新数据文本框
            self.data_text.delete(1.0, END)
            self.data_text.insert(END, f"📈 完整数据预览: {tensor_name}\n", 'data_header')
            self.data_text.insert(END, f"形状: {tensor_array.shape} | 总元素数: {len(flat_data)}\n\n", 'data_header')
            
            # 显示前100个值
            self.data_text.insert(END, "前100个元素值:\n", 'data_header')
            for i, value in enumerate(preview_data[:50]):  # 只显示前50个避免太长
                self.data_text.insert(END, f"[{i:5d}] ", 'data_index')
                self.data_text.insert(END, f"{value:.6f}\n", 'data_value')
            
            if len(preview_data) > 50:
                self.data_text.insert(END, f"... (共{len(preview_data)}个元素，仅显示前50个)\n", 'warning')
            
            # 更新滑块
            total_elements = len(flat_data)
            self.position_slider.config(to=total_elements - 1)
            self.max_position_label.config(text=str(total_elements - 1))
            self.data_range_label.config(text=f"范围: 0 - {total_elements - 1}")
            
        except Exception as e:
            self.data_text.delete(1.0, END)
            self.data_text.insert(END, f"❌ 加载数据失败: {str(e)}\n", 'data_error')
    
    def update_data_view(self, event=None):
        """更新数据视图"""
        if not hasattr(self, 'current_tensor_name') or not self.current_tensor_name:
            return
        
        try:
            position = self.position_var.get()
            page_size = self.page_size_var.get()
            page_size = max(10, min(page_size, 200))  # 限制每页大小
            
            tensor_array = self.load_tensor_data(self.current_tensor_name)
            if tensor_array is None:
                return
            
            flat_data = tensor_array.flatten()
            total_elements = len(flat_data)
            
            # 确保位置有效
            position = max(0, min(position, total_elements - 1))
            self.position_var.set(position)
            
            # 计算显示范围
            start_idx = position
            end_idx = min(start_idx + page_size, total_elements)
            
            # 更新数据文本框
            self.data_text.delete(1.0, END)
            self.data_text.insert(END, f"📊 数据查看: {self.current_tensor_name}\n", 'data_header')
            self.data_text.insert(END, f"当前位置: {start_idx} - {end_idx-1} | 总元素: {total_elements}\n\n", 'data_header')
            
            # 显示数据
            for i in range(start_idx, end_idx):
                if i >= total_elements:
                    break
                
                value = flat_data[i]
                self.data_text.insert(END, f"[{i:8d}] ", 'data_index')
                self.data_text.insert(END, f"{value:.6f}\n", 'data_value')
            
            # 高亮当前位置
            if start_idx < total_elements:
                line_start = f"{((start_idx - position) // page_size) + 2}.0"
                line_end = f"{((start_idx - position) // page_size) + 2}.end"
                self.data_text.tag_add('data_highlight', line_start, line_end)
            
        except Exception as e:
            self.data_text.delete(1.0, END)
            self.data_text.insert(END, f"❌ 显示数据失败: {str(e)}\n", 'data_error')
    
    def search_tensors(self, event=None):
        """搜索张量"""
        if not self.current_file_info:
            return
        
        query = self.search_var.get().strip().lower()
        if not query:
            return
        
        # 筛选匹配的张量
        filtered_tensors = []
        for tensor in self.current_file_info['tensors']:
            if 'error' in tensor:
                continue
            if query in tensor['name'].lower():
                filtered_tensors.append(tensor)
        
        # 更新树状结构（临时）
        self.tree.delete(*self.tree.get_children())
        
        search_node = self.tree.insert("", "end", "search_results", text=f"🔍 搜索结果: '{query}'", open=True)
        
        if not filtered_tensors:
            self.tree.insert(search_node, "end", "no_results", text="⚠️ 没有找到匹配的张量", values=('none', ''))
            self.status_var.set(f"🔍 搜索完成: 0 个匹配项")
            return
        
        for i, tensor in enumerate(filtered_tensors[:20]):  # 限制20个结果
            self.tree.insert(search_node, "end", tensor['name'], 
                           text=f"{tensor['name']} ({i+1}/{len(filtered_tensors)})",
                           values=(tensor['dtype'], str(tensor['shape'])))
        
        if len(filtered_tensors) > 20:
            self.tree.insert(search_node, "end", "more_results", 
                           text=f"... 共 {len(filtered_tensors)} 个匹配项，仅显示前20个", 
                           values=('info', ''))
        
        self.status_var.set(f"🔍 搜索完成: {len(filtered_tensors)} 个匹配项")
    
    def clear_search(self):
        """清除搜索"""
        self.search_var.set("")
        if self.current_file_info:
            # 重建原始树结构
            tensor_names = [t['name'] for t in self.current_file_info['tensors'] if 'error' not in t]
            structure = self.parse_model_structure(tensor_names)
            self.build_tree_structure(structure)
        self.status_var.set("搜索已清除")
    
    def show_component_info(self, component_name):
        """显示组件信息"""
        self.details_text.delete(1.0, END)
        
        info_map = {
            '词嵌入层': "🔤 词嵌入层 (embed_tokens)\n将token ID映射到向量空间，是模型的第一层。",
            'Transformer层': "🧱 Transformer层\n包含自注意力机制和前馈网络，是模型的核心组件。",
            '自注意力机制': "🎯 自注意力机制 (self_attn)\n- Q_proj: Query投影\n- K_proj: Key投影\n- V_proj: Value投影\n- O_proj: 输出投影\n- Q_norm/K_norm: 归一化层",
            'MLP前馈网络': "⚡ MLP前馈网络 (mlp)\n- gate_proj: 门控投影\n- up_proj: 上投影\n- down_proj: 下投影",
            'LayerNorm': "📊 LayerNorm层\n层归一化，稳定训练过程。",
            '最终归一化': "🎯 最终归一化 (norm)\nTransformer编码器的最后归一化层。",
            '语言模型头': "🎯 语言模型头 (lm_head)\n将隐藏状态映射回token ID空间。"
        }
        
        for key, info in info_map.items():
            if key in component_name:
                self.details_text.insert(END, f"📚 {component_name}\n", 'header')
                self.details_text.insert(END, "=" * 80 + "\n\n")
                self.details_text.insert(END, info + "\n")
                break
        else:
            self.details_text.insert(END, f"📚 {component_name}\n", 'header')
            self.details_text.insert(END, "=" * 80 + "\n\n")
            self.details_text.insert(END, "这是模型的一个组件，点击具体的张量查看详细信息。\n")
        
        self.details_text.insert(END, "\n💡 提示: 点击具体的张量（如权重）查看详细数据和完整向量。\n", 'warning')
        self.details_text.see(1.0)
    
    def export_to_json(self):
        """导出为JSON文件"""
        if not self.current_file_info:
            messagebox.showwarning("警告", "请先打开一个Safetensors文件")
            return
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            initialfile=f"{os.path.basename(self.current_file).replace('.safetensors', '')}_structure.json"
        )
        
        if not save_path:
            return
        
        try:
            # 构建层次化数据结构
            export_data = {
                'file_info': {
                    'path': self.current_file,
                    'size_mb': self.current_file_info['file_size'] / 1024 / 1024,
                    'tensor_count': len(self.current_file_info['tensors']),
                    'bfloat16_count': self.current_file_info['bfloat16_count'],
                    'torch_available': self.torch_available
                },
                'metadata': self.current_file_info['metadata'],
                'structure': {}
            }
            
            # 获取当前树状结构
            tensor_names = [t['name'] for t in self.current_file_info['tensors'] if 'error' not in t]
            structure = self.parse_model_structure(tensor_names)
            
            for layer_name, components in structure.items():
                layer_data = {}
                for comp_name, tensor_name in components.items():
                    # 查找张量信息
                    tensor_info = next((t for t in self.current_file_info['tensors'] if t['name'] == tensor_name), None)
                    if tensor_info:
                        layer_data[comp_name] = {
                            'tensor_name': tensor_name,
                            'shape': tensor_info['shape'],
                            'dtype': tensor_info['dtype'],
                            'size_bytes': tensor_info['size_bytes']
                        }
                export_data['structure'][layer_name] = layer_data
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.status_var.set(f"✅ 已导出到: {os.path.basename(save_path)}")
            messagebox.showinfo("成功", f"模型结构信息已导出到:\n{save_path}\n\n共导出 {len(export_data['structure'])} 个组件")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")
            self.status_var.set(f"❌ 导出失败: {str(e)}")
    
    def copy_content(self):
        """复制内容到剪贴板"""
        content = self.details_text.get(1.0, END)
        if content.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            self.status_var.set("✅ 详情内容已复制到剪贴板")
        else:
            self.status_var.set("⚠️ 详情内容为空")

def main():
    """主函数"""
    root = tk.Tk()
    
    # 设置主题
    style = ttk.Style()
    style.theme_use('clam')
    
    # 配置样式
    style.configure('TButton', font=('Arial', 10))
    style.configure('TLabel', font=('Arial', 10))
    style.configure('TRadiobutton', font=('Arial', 10))
    style.configure('TNotebook', background='#f0f2f5')
    style.configure('TNotebook.Tab', font=('Arial', 10, 'bold'))
    
    # 创建应用
    app = ModelHierarchyViewer(root)
    
    # 添加窗口关闭确认
    def on_closing():
        if messagebox.askokcancel("退出", "确定要退出程序吗？"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # 运行主循环
    root.mainloop()

if __name__ == "__main__":
    # 检查依赖
    try:
        from safetensors import safe_open
    except ImportError:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("错误", "缺少safetensors库。请运行:\npip install safetensors numpy torch")
        root.destroy()
        exit(1)
    
    try:
        import numpy as np
    except ImportError:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("错误", "缺少numpy库。请运行:\npip install numpy")
        root.destroy()
        exit(1)
    
    main()

