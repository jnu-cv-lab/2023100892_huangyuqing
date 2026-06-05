#!/usr/bin/env python3
"""
作业：实现并比较 Sinusoidal Position Encoding 与 RoPE

要求：
1. 实现 sinusoidal position encoding；
2. 实现二维向量旋转；
3. 实现高维 RoPE；
4. 对比 E+pos 和 RoPE 的输入方式；
5. 用数值实验验证 RoPE 的相对位置性质；
6. 说明：为什么 RoPE 比简单的 E+pos 更巧妙？
"""

import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. Sinusoidal Position Encoding =====================
class SinusoidalPositionEncoding:
    """原始 Transformer 使用的正余弦位置编码 (E+pos 方式)"""
    
    def __init__(self, d_model, max_len=5000):
        self.d_model = d_model
        self.max_len = max_len
        self.pe = self._generate_pe()
    
    def _generate_pe(self):
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(self.max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe
    
    def get_encoding(self, position):
        return self.pe[position]
    
    def apply_to_embedding(self, embedding, position):
        """加法注入位置信息 (E+pos)"""
        return embedding + self.get_encoding(position)


# ===================== 2. 二维向量旋转 =====================
class Rotator2D:
    @staticmethod
    def rotate_2d(vec, theta):
        """二维向量旋转"""
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
        if vec.ndim == 1:
            return rot_mat @ vec
        else:
            return vec @ rot_mat.T
    
    @staticmethod
    def rotation_matrix_2d(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])


# ===================== 3. 高维 RoPE =====================
class RotaryPositionEncoding:
    """旋转位置编码 (RoPE)"""
    
    def __init__(self, d_model, max_len=5000, base=10000.0):
        assert d_model % 2 == 0, "d_model 必须是偶数"
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # 每个维度对的频率
        self.freqs = 1.0 / (base ** (np.arange(0, d_model, 2) / d_model))
        # 预计算所有位置的 cos 和 sin
        self._precompute_angles(max_len)
    
    def _precompute_angles(self, max_len):
        positions = np.arange(max_len)
        angles = np.outer(positions, self.freqs)          # shape: (max_len, d_model//2)
        self.cos_cached = np.cos(angles)
        self.sin_cached = np.sin(angles)
    
    def apply_rope(self, x, pos):
        """
        对张量 x 应用 RoPE
        x: shape (..., d_model)
        pos: 位置索引（非负整数）
        """
        d = self.d_model // 2
        # 取对应位置的 cos/sin
        cos = self.cos_cached[pos]   # shape: (d,)
        sin = self.sin_cached[pos]
        # 广播维度（如果 x 有 batch 维）
        while cos.ndim < x.ndim:
            cos = np.expand_dims(cos, axis=0)
            sin = np.expand_dims(sin, axis=0)
        # 拆分两半
        x1, x2 = x[..., :d], x[..., d:]
        # 旋转公式
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        return np.concatenate([x1_rot, x2_rot], axis=-1)
    
    def get_relative_embedding(self, q, k, pos_q, pos_k):
        """返回 RoPE(q,pos_q) 与 RoPE(k,pos_k) 的内积"""
        q_rot = self.apply_rope(q, pos_q)
        k_rot = self.apply_rope(k, pos_k)
        return np.dot(q_rot, k_rot)
    
    def relative_property_check(self, q, k, pos_q, pos_k):
        """
        验证核心性质：
        <RoPE(q,pos_q), RoPE(k,pos_k)> = <q, RoPE(k, pos_k - pos_q)>
        注意：这里要求 pos_k >= pos_q，否则使用绝对值会丢失符号信息，
        但为了演示，我们只验证正相对位置。
        """
        assert pos_k >= pos_q, "验证时请确保 pos_k >= pos_q"
        rel = pos_k - pos_q
        left = self.get_relative_embedding(q, k, pos_q, pos_k)
        k_rot_rel = self.apply_rope(k, rel)
        right = np.dot(q, k_rot_rel)
        return left, right


# ===================== 对比实验 =====================
class PositionEncodingComparison:
    def __init__(self, d_model=64, seq_len=20):
        self.d_model = d_model
        self.seq_len = seq_len
        self.sin_pe = SinusoidalPositionEncoding(d_model, max_len=seq_len*2)
        self.rope = RotaryPositionEncoding(d_model, max_len=seq_len*2)
    
    def demo_2d_rotation(self):
        """要求2：演示二维向量旋转"""
        print("\n" + "="*50)
        print("2. 二维向量旋转演示")
        print("="*50)
        rot = Rotator2D()
        vec = np.array([1.0, 0.0])
        angles = [0, np.pi/4, np.pi/2, np.pi]
        names = ["0°", "45°", "90°", "180°"]
        
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for ax, theta, name in zip(axes, angles, names):
            v_rot = rot.rotate_2d(vec, theta)
            ax.arrow(0, 0, vec[0], vec[1], head_width=0.1, color='b', label='original')
            ax.arrow(0, 0, v_rot[0], v_rot[1], head_width=0.1, color='r', label='rotated')
            ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
            ax.set_title(f"旋转 {name}"); ax.grid(True); ax.legend()
            print(f"旋转{name}: {vec} -> {v_rot}")
        plt.tight_layout()
        plt.savefig("2d_rotation.png")
        plt.show()
    
    def demo_sinusoidal_encoding(self):
        """要求1：展示 Sinusoidal PE 的形状"""
        print("\n" + "="*50)
        print("1. Sinusoidal Position Encoding 可视化")
        print("="*50)
        pe = self.sin_pe.pe[:self.seq_len, :8]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        im = ax1.imshow(pe.T, aspect='auto', cmap='RdBu')
        ax1.set_xlabel("Position"); ax1.set_ylabel("Dimension"); ax1.set_title("Sinusoidal PE (前8维)")
        plt.colorbar(im, ax=ax1)
        for dim in range(4):
            ax2.plot(pe[:, dim], label=f"dim{dim}")
        ax2.set_xlabel("Position"); ax2.set_ylabel("Value"); ax2.set_title("不同维度的编码值")
        ax2.legend(); ax2.grid(True)
        plt.tight_layout()
        plt.savefig("sinusoidal_pe.png")
        plt.show()
        print(f"位置0的编码(前8维): {self.sin_pe.get_encoding(0)[:8]}")
        print(f"位置1的编码(前8维): {self.sin_pe.get_encoding(1)[:8]}")
    
    def demo_rope_relative_property(self):
        """要求5：用数值实验验证 RoPE 的相对位置性质"""
        print("\n" + "="*50)
        print("5. 验证 RoPE 的相对位置性质")
        print("="*50)
        np.random.seed(42)
        q = np.random.randn(self.d_model)
        k = np.random.randn(self.d_model)
        q = q / np.linalg.norm(q)
        k = k / np.linalg.norm(k)
        
        print("核心性质: <RoPE(q,m), RoPE(k,n)> = <q, RoPE(k, n-m)>\n")
        test_pairs = [(0,1), (1,2), (0,3), (2,5)]   # 保证 n > m
        for m, n in test_pairs:
            left, right = self.rope.relative_property_check(q, k, m, n)
            print(f"m={m}, n={n}, rel={n-m}:")
            print(f"  左式 = {left:.6f}")
            print(f"  右式 = {right:.6f}")
            print(f"  误差 = {abs(left-right):.2e}\n")
        
        # 绘制内积随相对位置的变化图
        positions = list(range(-self.seq_len//2, self.seq_len//2))
        scores = []
        for rel in positions:
            if rel >= 0:
                k_rot = self.rope.apply_rope(k, rel)
            else:
                # 对于负相对位置，等价于反向旋转（这里用 abs 并取负号？为简单只演示正侧）
                # 注意：严格 RoPE 对于负相对位置，内积 = <RoPE(q,0), RoPE(k, -rel)> 的某些形式
                # 这里只演示 |rel| 对应的值（取绝对值，对称性由 cosine 偶函数保证）
                k_rot = self.rope.apply_rope(k, -rel)
            score = np.dot(q, k_rot)
            scores.append(score)
        plt.figure(figsize=(8,5))
        plt.plot(positions, scores, 'bo-', markersize=3)
        plt.xlabel("相对位置 (key 位置 - query 位置)")
        plt.ylabel("内积值")
        plt.title("RoPE: 内积随相对位置的变化")
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig("rope_relative_property.png")
        plt.show()
    
    def compare_attention_patterns(self):
        """要求4：对比 E+pos 和 RoPE 的输入方式引起的注意力模式差异"""
        print("\n" + "="*50)
        print("4. 对比 E+pos vs RoPE 的注意力模式")
        print("="*50)
        np.random.seed(42)
        embeddings = np.random.randn(self.seq_len, self.d_model)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # E+pos: 加法注入
        epos_emb = np.array([self.sin_pe.apply_to_embedding(embeddings[pos], pos) 
                             for pos in range(self.seq_len)])
        # RoPE: 旋转注入
        rope_emb = np.array([self.rope.apply_rope(embeddings[pos], pos) 
                             for pos in range(self.seq_len)])
        
        att_epos = epos_emb @ epos_emb.T
        att_rope = rope_emb @ rope_emb.T
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        im1 = axes[0].imshow(att_epos, cmap='RdBu', aspect='auto')
        axes[0].set_title("E+pos (加法位置编码)")
        axes[0].set_xlabel("Key 位置"); axes[0].set_ylabel("Query 位置")
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(att_rope, cmap='RdBu', aspect='auto')
        axes[1].set_title("RoPE (旋转位置编码)")
        axes[1].set_xlabel("Key 位置"); axes[1].set_ylabel("Query 位置")
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(att_rope - att_epos, cmap='RdBu', aspect='auto')
        axes[2].set_title("差异 (RoPE - E+pos)")
        axes[2].set_xlabel("Key 位置"); axes[2].set_ylabel("Query 位置")
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig("attention_comparison.png")
        plt.show()
        
        # 打印对角线附近的平均值
        print("对角线附近注意力值平均 (offset 1~4):")
        for offset in range(1, 5):
            epos_mean = np.mean([att_epos[i, i+offset] for i in range(self.seq_len-offset)])
            rope_mean = np.mean([att_rope[i, i+offset] for i in range(self.seq_len-offset)])
            print(f"  offset={offset}: E+pos={epos_mean:.4f}, RoPE={rope_mean:.4f}")
    
    def run_all(self):
        self.demo_2d_rotation()
        self.demo_sinusoidal_encoding()
        self.demo_rope_relative_property()
        self.compare_attention_patterns()
        self.explain_why_rope_better()
    
    def explain_why_rope_better(self):
        """要求6：说明 RoPE 更巧妙的原因"""
        print("\n" + "="*50)
        print("6. 为什么 RoPE 比简单的 E+pos 更巧妙？")
        print("="*50)
        explanation = """
        【核心优势】
        1. 显式相对位置建模
           - RoPE 通过旋转矩阵直接编码相对位置，内积结果只依赖于相对距离。
           - 而 E+pos 的绝对位置相加，内积中会混有绝对位置的交叉项，难以解耦。
        
        2. 优秀的长度外推能力
           - 旋转频率与位置解耦，训练时未见过的更长序列也能合理计算相对位置注意力。
           - E+pos 的 sin/cos 函数虽然也能外推，但绝对位置相加会导致外推时注意力分布畸变。
        
        3. 保持向量模长不变
           - 旋转是正交变换，RoPE 不改变向量的模长，避免了数值不稳定。
           - E+pos 直接相加会改变 embedding 的模长，可能影响训练稳定性。
        
        4. 不引入额外参数
           - RoPE 不需要学习任何参数，完全由数学公式定义。
           - 相比可学习的位置编码，RoPE 参数效率更高，且不易过拟合。
        
        5. 数学优雅性
           - 满足严格的数学性质: ⟨RoPE(q,m), RoPE(k,n)⟩ = ⟨q, RoPE(k, n-m)⟩。
           - 这种性质使得模型可以自动关注相对位置，无需显式特征工程。
        
        【总结】
        RoPE 从“旋转”这一几何变换出发，将位置信息自然地融入向量内积，
        既保持了相对位置的优势，又具有绝对位置的实现便捷性，是位置编码的重要革新。
        """
        print(explanation)


# ===================== 主程序 =====================
def main():
    print("\n" + "="*60)
    print("作业：实现并比较 Sinusoidal Position Encoding 与 RoPE")
    print("="*60)
    comp = PositionEncodingComparison(d_model=64, seq_len=20)
    comp.run_all()
    print("\n所有实验完成！生成的图片已保存到当前目录。")

if __name__ == "__main__":
    main()