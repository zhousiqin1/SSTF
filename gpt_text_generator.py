import openai
import time
import logging
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
from scipy import signal
from fallback_text_generator import FallbackTextGenerator
from simple_tokenizer import SimpleTokenizer
from text_utils import save_texts_to_csv, load_texts_from_csv

class GPTTextGenerator:
    def __init__(self, args, clip_model=None):
        """
        初始化文本生成器
        
        Args:
            args: 配置参数（来自config.py的parse_args()）
            clip_model: 可选，CLIP模型对象，用于动态token数估算
        """
        # 修正getattr参数
        self.api_key = getattr(args, 'openai_api_key', None)
        self.model = getattr(args, 'gpt_model', 'gpt-3.5-turbo')
        self.use_gpt = getattr(args, 'use_gpt', False)
        self.max_tokens = getattr(args, 'gpt_max_tokens', 70)
        self.temperature = getattr(args, 'gpt_temperature', 0.5)
        self.timeout = getattr(args, 'gpt_timeout', 30)
        self.max_retries = getattr(args, 'gpt_max_retries', 3)
        self.system_prompt = getattr(args, 'gpt_system_prompt', "")
        self.logger = logging.getLogger(__name__)
        self.base_url = getattr(args, 'gpt_base_url', None)
        
        # 初始化tokenizer用于精确token计数（使用BPE分词）
        self.tokenizer = SimpleTokenizer(bpe_path="bpe_simple_vocab_16e6.txt.gz")
        
        # 初始化普通文本生成器
        self.fallback_generator = FallbackTextGenerator(max_tokens=self.max_tokens)
        
        # 根据模式进行初始化
        if self.use_gpt:
            # GPT模式
            if self.api_key:
                if self.base_url:
                    self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
                else:
                    self.client = openai.OpenAI(api_key=self.api_key)
                self.logger.info(f"Using GPT mode, model: {self.model}, max tokens: {self.max_tokens}, base_url: {self.base_url}")
            else:
                self.logger.warning("GPT mode requires API key, switching to normal mode")
                self.use_gpt = False
        else:
            # 普通模式
            self.logger.info(f"Using rule-based text generation mode, max tokens: {self.max_tokens}")
        
        self.clip_model = clip_model

        # 特征统计信息存储
        self.feature_stats = {}
        self.stats_computed = False
        

    def extract_enhanced_features(self, window_data: np.ndarray) -> Dict[str, float]:
        """
        提取7个关键特征（专门针对反刍、进食、其他三种行为优化）
        
        Args:
            window_data: 传感器数据窗口
            
        Returns:
            7个关键特征字典：periodicity, pitch_mean, pitch_std, vm_mean, vm_std, complexity, zero_crossings
        """
        if window_data.ndim == 1:
            window_data = window_data.reshape(-1, 1)
        # 分离三轴数据
        x_data = window_data[:, 0]  # 前后方向
        y_data = window_data[:, 1]  # 左右方向
        z_data = window_data[:, 2]  # 上下方向
        
        # 计算合成加速度
        vm = np.sqrt(x_data**2 + y_data**2 + z_data**2)
        
        # 1. 周期性特征（反刍的关键特征）
        # 自相关分析
        autocorr = np.correlate(vm, vm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr_peaks, _ = signal.find_peaks(autocorr[:len(autocorr)//2])
        autocorr_peak_count = len(autocorr_peaks)
        
        # 周期性强度
        if len(autocorr) > 1:
            periodicity = np.max(autocorr[1:]) / autocorr[0]
        else:
            periodicity = 0
        
        # 2. 头部姿态特征（进食的关键特征）
        # 俯仰角（前后倾斜）
        pitch_angles = np.arctan2(x_data, np.sqrt(y_data**2 + z_data**2))
        pitch_mean = np.mean(pitch_angles)
        pitch_std = np.std(pitch_angles)
            
        # 横滚角（左右倾斜）
        roll_angles = np.arctan2(y_data, np.sqrt(x_data**2 + z_data**2))
        roll_mean = np.mean(roll_angles)
        roll_std = np.std(roll_angles)
            
        # 3. 运动强度特征（其他活动的关键特征）
        vm_mean = np.mean(vm)
        vm_std = np.std(vm)
        energy = np.sum(vm**2)
        
        # 4. 时序规律性特征
        peaks, _ = signal.find_peaks(vm)
        peak_count = len(peaks)
        
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            avg_peak_interval = np.mean(peak_intervals)
        else:
            avg_peak_interval = len(vm)
        
        # 5. 运动复杂度特征
        # 基于熵的复杂度
        hist, _ = np.histogram(vm, bins=20)
        # 归一化为概率分布
        prob = hist / np.sum(hist)
        # 只对非零概率计算熵，避免log(0)
        prob = prob[prob > 0]
        if len(prob) > 0:
            complexity = -np.sum(prob * np.log(prob))
        else:
            complexity = 0
        
        # 过零点数
        zero_crossings = np.sum(np.diff(np.signbit(vm - np.mean(vm))) != 0)
        
        return {
            # 周期性特征（反刍关键）- 最重要的反刍指标
            'periodicity': periodicity,
            
            # 头部姿态特征（进食关键）- 最重要的进食指标
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            
            # 运动强度特征（其他活动关键）- 区分活动强度
            'vm_mean': vm_mean,
            'vm_std': vm_std,
            
            # 运动复杂度特征 - 区分规律性
            'complexity': complexity,
            
            # 过零点数 - 反刍的重复性指标
            'zero_crossings': zero_crossings
        }

    def compute_feature_statistics(self, all_features_list):
        """
        计算特征的统计分布，用于动态阈值

        Args:
            all_features_list: 所有窗口的特征字典列表
        """
        if not all_features_list:
            return

        # 计算关键特征的统计分布（只保留最重要的7个特征）
        key_features = ['periodicity', 'pitch_mean', 'pitch_std', 'vm_mean', 'vm_std', 'complexity', 'zero_crossings']

        for feature_name in key_features:
            values = [f[feature_name] for f in all_features_list if feature_name in f]
            if values:
                self.feature_stats[feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'p25': np.percentile(values, 25),
                    'p50': np.percentile(values, 50),
                    'p75': np.percentile(values, 75),
                    'p80': np.percentile(values, 80),
                    'p85': np.percentile(values, 85),
                    'p90': np.percentile(values, 90)
                }

        self.stats_computed = True
        print(f"特征统计计算完成，涵盖 {len(key_features)} 个关键特征")


    # 标签数字到行为名称的映射
    LABEL_MAP = {
        0: "Other",
        1: "Rumination",
        2: "Feeding"
    }



    def generate_enhanced_prompt(self, features: Dict[str, float], label: int, 
                                behavior_type: str, window_data: np.ndarray, 
                                window_labels: np.ndarray) -> str:
        """
        生成优化的Prompt
        
        Args:
            features: 特征字典
            label: 行为标签
            behavior_type: 行为类型
            window_data: 窗口数据
            window_labels: 窗口标签
            
        Returns:
            优化的Prompt字符串
        """
        # 直接生成数据摘要，使用7个关键特征
        data_summary = self._generate_data_summary(window_data, features, window_labels)
        # 统计窗口主标签和分布
        if window_labels is not None and len(window_labels) > 0:
            label_counter = Counter(window_labels)
            main_label_num, main_label_count = label_counter.most_common(1)[0]
            main_label_name = self.LABEL_MAP.get(main_label_num, str(main_label_num))
            total = len(window_labels)
            dist_str = ", ".join([
                f"{self.LABEL_MAP.get(lbl, str(lbl))}({cnt/total:.0%})"
                for lbl, cnt in label_counter.items()
            ])
            label_info = f"Window main label: {main_label_name} ({main_label_num}); Distribution: {dist_str}"
        else:
            label_info = "Window main label: Unknown; Distribution: Unknown"
        # 传递给下一级
        prompt = self._generate_optimized_prompt(label_info, data_summary, features, window_labels)
        return prompt.strip()



    def _generate_data_summary(self, window_data: np.ndarray, features: Dict[str, float],
                             window_labels: np.ndarray) -> str:
        """生成行为区分性摘要 - 包含7个关键特征让GPT全面分析"""
        # 提取7个关键特征
        periodicity = features.get('periodicity', 0)
        pitch_mean = features.get('pitch_mean', 0)
        pitch_std = features.get('pitch_std', 0)
        vm_mean = features.get('vm_mean', 0)
        vm_std = features.get('vm_std', 0)
        complexity = features.get('complexity', 0)
        zero_crossings = features.get('zero_crossings', 0)

        enhanced_indicators = []

        # 收集所有特征描述，然后智能选择最重要的
        feature_descriptions = []

        # 1. 周期性特征描述（反刍的关键指标）
        if self.stats_computed and 'periodicity' in self.feature_stats:
            stats = self.feature_stats['periodicity']
            if periodicity > stats['p90']:
                feature_descriptions.append((f"very strong rhythmic pattern (periodicity={periodicity:.3f})", 4))
            elif periodicity > stats['p80']:
                feature_descriptions.append((f"strong rhythmic pattern (periodicity={periodicity:.3f})", 3))
            elif periodicity > stats['p50']:
                feature_descriptions.append((f"moderate rhythmic pattern (periodicity={periodicity:.3f})", 2))
            elif periodicity > stats['p25']:
                feature_descriptions.append((f"weak rhythmic pattern (periodicity={periodicity:.3f})", 1))
        else:
            # 回退到固定阈值
            if periodicity > 0.6:
                feature_descriptions.append((f"very strong rhythmic pattern (periodicity={periodicity:.3f})", 4))
            elif periodicity > 0.4:
                feature_descriptions.append((f"strong rhythmic pattern (periodicity={periodicity:.3f})", 3))
            elif periodicity > 0.2:
                feature_descriptions.append((f"moderate rhythmic pattern (periodicity={periodicity:.3f})", 2))
            elif periodicity > 0:
                feature_descriptions.append((f"weak rhythmic pattern (periodicity={periodicity:.3f})", 1))

        # 2. 头部姿态描述（进食的关键指标）
        if abs(pitch_mean) > 0:
            pitch_normalized = abs(pitch_mean) / (pitch_std + 1e-6)
            if pitch_normalized > 2.0:
                if pitch_mean < 0:
                    feature_descriptions.append((f"significant head down (pitch={pitch_mean:.3f})", 4))
                else:
                    feature_descriptions.append((f"significant head up (pitch={pitch_mean:.3f})", 4))
            elif pitch_normalized > 1.0:
                if pitch_mean < 0:
                    feature_descriptions.append((f"moderate head down (pitch={pitch_mean:.3f})", 3))
                else:
                    feature_descriptions.append((f"moderate head up (pitch={pitch_mean:.3f})", 3))

        # 3. 头部姿态稳定性描述
        if pitch_std > 0:
            if self.stats_computed and 'pitch_std' in self.feature_stats:
                stats = self.feature_stats['pitch_std']
                if pitch_std > stats['p85']:
                    feature_descriptions.append((f"variable head posture (pitch_std={pitch_std:.3f})", 2))
                elif pitch_std > stats['p50']:
                    feature_descriptions.append((f"stable head posture (pitch_std={pitch_std:.3f})", 1))
            else:
                # 回退到固定阈值
                if pitch_std > 0.5:
                    feature_descriptions.append((f"variable head posture (pitch_std={pitch_std:.3f})", 2))
                elif pitch_std > 0.2:
                    feature_descriptions.append((f"stable head posture (pitch_std={pitch_std:.3f})", 1))

        # 4. 运动强度描述（其他活动的关键指标）
        if self.stats_computed and 'vm_mean' in self.feature_stats:
            stats = self.feature_stats['vm_mean']
            if vm_mean > stats['p85']:
                feature_descriptions.append((f"high intensity (vm={vm_mean:.3f})", 4))
            elif vm_mean > stats['p50']:
                feature_descriptions.append((f"moderate intensity (vm={vm_mean:.3f})", 3))
            elif vm_mean > 0:
                feature_descriptions.append((f"low intensity (vm={vm_mean:.3f})", 2))
        else:
            # 回退到固定阈值
            if vm_mean > 1.5:
                feature_descriptions.append((f"high intensity (vm={vm_mean:.3f})", 4))
            elif vm_mean > 0.8:
                feature_descriptions.append((f"moderate intensity (vm={vm_mean:.3f})", 3))
            elif vm_mean > 0:
                feature_descriptions.append((f"low intensity (vm={vm_mean:.3f})", 2))

        # 5. 运动强度变化性描述
        if vm_std > 0:
            if self.stats_computed and 'vm_std' in self.feature_stats:
                stats = self.feature_stats['vm_std']
                if vm_std > stats['p85']:
                    feature_descriptions.append((f"highly variable movement (vm_std={vm_std:.3f})", 3))
                elif vm_std > stats['p50']:
                    feature_descriptions.append((f"variable movement (vm_std={vm_std:.3f})", 2))
                else:
                    feature_descriptions.append((f"steady movement (vm_std={vm_std:.3f})", 1))
            else:
                # 回退到固定阈值
                if vm_std > 0.8:
                    feature_descriptions.append((f"highly variable movement (vm_std={vm_std:.3f})", 3))
                elif vm_std > 0.4:
                    feature_descriptions.append((f"variable movement (vm_std={vm_std:.3f})", 2))
                elif vm_std > 0.1:
                    feature_descriptions.append((f"steady movement (vm_std={vm_std:.3f})", 1))

        # 6. 运动复杂度描述（区分规律性）
        if self.stats_computed and 'complexity' in self.feature_stats:
            stats = self.feature_stats['complexity']
            if complexity > stats['p85']:
                feature_descriptions.append((f"very complex pattern (complexity={complexity:.3f})", 3))
            elif complexity > stats['p50']:
                feature_descriptions.append((f"complex pattern (complexity={complexity:.3f})", 2))
            elif complexity > 0:
                feature_descriptions.append((f"simple pattern (complexity={complexity:.3f})", 1))
        else:
            # 回退到固定阈值
            if complexity > 3.0:
                feature_descriptions.append((f"very complex pattern (complexity={complexity:.3f})", 3))
            elif complexity > 2.0:
                feature_descriptions.append((f"complex pattern (complexity={complexity:.3f})", 2))
            elif complexity > 0:
                feature_descriptions.append((f"simple pattern (complexity={complexity:.3f})", 1))

        # 7. 过零点数描述（反刍的重复性指标）
        if zero_crossings > 0:
            if self.stats_computed and 'zero_crossings' in self.feature_stats:
                stats = self.feature_stats['zero_crossings']
                if zero_crossings > stats['p85']:
                    feature_descriptions.append((f"high repetitive motion (zero_crossings={zero_crossings:.0f})", 3))
                elif zero_crossings > stats['p50']:
                    feature_descriptions.append((f"moderate repetitive motion (zero_crossings={zero_crossings:.0f})", 2))
                else:
                    feature_descriptions.append((f"low repetitive motion (zero_crossings={zero_crossings:.0f})", 1))
            else:
                # 回退到固定阈值
                if zero_crossings > 50:
                    feature_descriptions.append((f"high repetitive motion (zero_crossings={zero_crossings:.0f})", 3))
                elif zero_crossings > 20:
                    feature_descriptions.append((f"moderate repetitive motion (zero_crossings={zero_crossings:.0f})", 2))
                elif zero_crossings > 5:
                    feature_descriptions.append((f"low repetitive motion (zero_crossings={zero_crossings:.0f})", 1))

        # 将所有7个关键特征描述发送给GPT，让GPT自己分析选择
        # 按重要性评分排序，但保留所有特征
        feature_descriptions.sort(key=lambda x: x[1], reverse=True)
        all_indicators = [desc for desc, score in feature_descriptions]

        # 生成完整的特征摘要，包含所有特征信息
        if all_indicators:
            summary = "; ".join(all_indicators)
        else:
            # 如果没有特征描述，提供基本的运动信息（7个关键特征）
            basic_info = []
            if vm_mean > 0:
                basic_info.append(f"movement detected (vm_mean={vm_mean:.3f})")
            if periodicity > 0:
                basic_info.append(f"periodicity={periodicity:.3f}")
            if abs(pitch_mean) > 0:
                basic_info.append(f"head angle (pitch={pitch_mean:.3f})")
            if pitch_std > 0:
                basic_info.append(f"head stability (pitch_std={pitch_std:.3f})")
            if vm_std > 0:
                basic_info.append(f"movement variability (vm_std={vm_std:.3f})")
            if complexity > 0:
                basic_info.append(f"complexity={complexity:.3f}")
            if zero_crossings > 0:
                basic_info.append(f"zero_crossings={zero_crossings:.0f}")
            
            summary = "; ".join(basic_info) if basic_info else "minimal movement detected"
        
        return summary


    def _generate_optimized_prompt(self, label_info: str, data_summary: str, features: Dict[str, float], 
                                 window_labels: np.ndarray = None) -> str:
        import random
        
        # 去除上下文处理，专注于当前窗口的特征分析
        
        # 根据特征提供相关示例
        examples = self._generate_dynamic_examples(features, window_labels)
        
        # 生成准确性指导
        accuracy_guidance = self._generate_creativity_guidance()
        
        prompt = f'''
{examples}

{accuracy_guidance}

{label_info}

Movement features analysis:
{data_summary}

Based on the movement features above, generate a natural behavioral description (8-12 words) that:
- Includes the behavior type name (feeding behavior/rumination behavior/other behavior) naturally in the sentence
- Focuses on the most distinctive movement characteristics from the analysis
- Uses simple, clear vocabulary - avoid complex or technical terms
- Keep the language straightforward and easy to understand
- Use direct, simple sentences instead of complex structures
- Let the natural flow determine where the behavior name appears
- Follow the style and length of the reference example but create your own unique description
- Do NOT include quotation marks or any formatting symbols
- Generate ONLY the text description, nothing else
- Output format: plain text only, no quotes, no brackets, no symbols

Behavioral description:'''
        return prompt

    def _generate_dynamic_examples(self, features: Dict[str, float], window_labels: np.ndarray = None) -> str:
        """根据当前窗口标签提供相关示例"""
        import random
        from collections import Counter
        
        # 根据标签选择示例类型
        if window_labels is not None and len(window_labels) > 0:
            # 使用窗口的主要标签（出现最多的标签）
            label_counter = Counter(window_labels)
            current_label = label_counter.most_common(1)[0][0]  # 获取出现最多的标签
            
            if current_label == 0:  # other
                selected_type = 'other'
            elif current_label == 1:  # rumination
                selected_type = 'rumination'
            elif current_label == 2:  # feeding
                selected_type = 'feeding'
            else:  # 未知标签，随机选择
                selected_type = random.choice(['feeding', 'rumination', 'other'])
        else:
            # 如果没有标签信息，随机选择
            selected_type = random.choice(['feeding', 'rumination', 'other'])
        
        # 8-12个单词的示例
        example_pools = {
            'rumination': [
                "Strong rhythmic chewing patterns during rumination activity",
                "Consistent jaw movements with low intensity during rumination",
                "Regular chewing rhythm with steady pattern during rumination"
            ],
            'feeding': [
                "Downward head posture with steady feeding movements",
                "Low head position during feeding with moderate intensity",
                "Head-down grazing motion with steady rhythm during feeding"
            ],
            'other': [
                "Irregular motion patterns with variable intensity in other activity",
                "Mixed movement patterns with high changes in other behavior",
                "Unpredictable motion patterns with moderate intensity in other activity"
            ]
        }
        
        selected_example = random.choice(example_pools[selected_type])
        
        return f"Style reference: {selected_example}\n\nGenerate your own description (8-12 words). Output format: plain text only, no quotes, no symbols:\n"

    def _generate_creativity_guidance(self) -> str:
        """生成平衡的多样性和一致性指导"""
        import random
        
        guidance_options = [
            "Create unique descriptions while maintaining consistency in style and format.",
            "Use varied vocabulary but keep the same general structure as the reference.",
            "Be creative with word choices while staying accurate and consistent.",
            "Generate diverse expressions within a consistent descriptive framework."
        ]
        
        return random.choice(guidance_options)



    def call_gpt_api(self, prompt: str, window_data: np.ndarray, window_labels: np.ndarray, features: Dict[str, float] = None) -> str:
        """
        调用GPT API生成文本
        """
        max_bpe_tokens = 70
        for attempt in range(self.max_retries):
            try:
                # 动态调整max_tokens
                dynamic_max_tokens = min(self.max_tokens, 70)
                if hasattr(self, 'clip_model') and self.clip_model is not None:
                    try:
                        import clip
                        base_tokens = clip.tokenize([prompt])[0]
                        base_token_count = (base_tokens != 0).sum().item()
                        dynamic_max_tokens = max(10, 77 - base_token_count - 5)
                    except Exception as e:
                        pass
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=dynamic_max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout
                )
                generated_text = response.choices[0].message.content.strip()
                # 检查BPE分词后token数，超限则重试
                bpe_tokens = self.tokenizer.encode(generated_text)
                if len(bpe_tokens) <= max_bpe_tokens:
                    return generated_text
                else:
                    self.logger.warning(f"Generated text BPE tokens {len(bpe_tokens)} > {max_bpe_tokens}, retrying...")
            except Exception as e:
                print(f"API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    self.logger.warning(f"GPT API call failed, falling back to fallback generator")
                    return self.fallback_generator.generate_text_for_window(window_data, window_labels)
        # 多次重试后仍超限，降级到fallback生成器
        self.logger.error(f"Failed to generate text within {max_bpe_tokens} BPE tokens after {self.max_retries} attempts. Falling back to fallback generator.")
        return self.fallback_generator.generate_text_for_window(window_data, window_labels)
    
    def generate_text_for_window(self, window_data: np.ndarray, window_labels: np.ndarray) -> str:
        """
        为单个窗口生成文本描述
        
        Args:
            window_data: 传感器数据窗口 [window_size, 3]
            window_labels: 窗口内所有标签 [window_size]
            
        Returns:
            生成的文本描述
        """
        if self.use_gpt:
            # GPT模式：使用GPT API生成文本
            return self._generate_gpt_text(window_data, window_labels)
        else:
            # 普通模式：使用基于规则的文本生成器
            return self.fallback_generator.generate_text_for_window(window_data, window_labels)
    
    def _generate_gpt_text(self, window_data: np.ndarray, window_labels: np.ndarray) -> str:
        """
        使用GPT API生成文本描述
        
        Args:
            window_data: 传感器数据窗口
            window_labels: 窗口标签
            
        Returns:
            GPT生成的文本描述
        """
        # 提取增强的窗口特征
        features = self.extract_enhanced_features(window_data)
        
        # 如果还没有计算特征统计，使用当前窗口特征进行简单统计
        if not self.stats_computed:
            print("单窗口模式：使用当前特征进行简单统计计算...")
            self.compute_feature_statistics([features])

        # 使用统一的Prompt生成
        prompt = self.generate_enhanced_prompt(features, 0, "", window_data, window_labels)

        # 调用GPT API，传递features避免重复提取
        text_description = self.call_gpt_api(prompt, window_data, window_labels, features)
        
        return text_description

    def generate_text_for_window_with_features(self, window_data: np.ndarray, window_labels: np.ndarray, cached_features: Dict[str, float]) -> str:
        """
        使用缓存的特征为窗口生成文本描述（避免重复特征提取）

        Args:
            window_data: 传感器数据窗口
            window_labels: 窗口标签
            cached_features: 预先计算的特征

        Returns:
            生成的文本描述
        """
        if self.use_gpt:
            # 如果还没有计算特征统计，使用当前窗口特征进行简单统计
            if not self.stats_computed:
                print("单窗口模式（缓存特征）：使用当前特征进行简单统计计算...")
                self.compute_feature_statistics([cached_features])
            
            # GPT模式：使用缓存的特征
            prompt = self.generate_enhanced_prompt(cached_features, 0, "", window_data, window_labels)
            return self.call_gpt_api(prompt, window_data, window_labels, cached_features)
        else:
            # 普通模式：使用基于规则的文本生成器
            return self.fallback_generator.generate_text_for_window(window_data, window_labels)

    def generate_texts_for_windows_batch(self, features: np.ndarray, labels: np.ndarray,
                                       window_size: int, stride: int, batch_size: int = 32,
                                       save_path: str = None) -> List[str]:
        """
        分批为所有窗口生成文本描述（内存优化版本）
        
        Args:
            features: 传感器数据
            labels: 标签数据
            window_size: 窗口大小
            stride: 窗口步长
            batch_size: 批处理大小
            save_path: 保存路径（可选）
            
        Returns:
            文本描述列表
        """
        text_descriptions = []

        if self.use_gpt:
            print(f"Starting batch generation of window text descriptions using {self.model}...")
            print("Mode: GPT API generation (batch processing)")
        else:
            print("Starting batch generation of window text descriptions using rule-based text generator...")
            print("Mode: Rule-based generation (batch processing)")
        print(f"Window size: {window_size}, Stride: {stride}, Batch size: {batch_size}")

        # 计算窗口数量
        num_windows = (len(features) - window_size) // stride + 1
        print(f"Total windows: {num_windows}")

        # 预先计算所有窗口的特征并缓存（避免重复计算）
        print("提取所有窗口特征...")
        all_features = {}  # 缓存所有窗口的特征
        for window_idx in range(num_windows):
            start_idx = window_idx * stride
            end_idx = start_idx + window_size
            window_data = features[start_idx:end_idx]
            window_features = self.extract_enhanced_features(window_data)
            all_features[window_idx] = window_features

        # 计算特征统计分布（用于动态阈值）
        if not self.stats_computed:
            print("计算特征统计分布...")
            self.compute_feature_statistics(list(all_features.values()))
            print("特征统计计算完成")
        
        # 分批处理
        for batch_start in range(0, num_windows, batch_size):
            batch_end = min(batch_start + batch_size, num_windows)
            batch_texts = []
            
            print(f"Processing text batch {batch_start//batch_size + 1}/{(num_windows + batch_size - 1)//batch_size}")
            
            for window_idx in range(batch_start, batch_end):
                start_idx = window_idx * stride
                end_idx = start_idx + window_size

                # 获取窗口数据
                window_data = features[start_idx:end_idx]
                window_labels = labels[start_idx:end_idx]

                # 使用缓存的特征生成文本描述
                cached_features = all_features[window_idx]
                text = self.generate_text_for_window_with_features(window_data, window_labels, cached_features)
                batch_texts.append(text)
                
                # 只在GPT模式下添加延迟（避免API限制）
                if self.use_gpt:
                    time.sleep(0.1)
            
            # 添加到总列表
            text_descriptions.extend(batch_texts)
            
            # 清理内存
            del batch_texts
            import gc
            gc.collect()
        
        print(f"Window text generation completed, generated {len(text_descriptions)} descriptions")
        
        # 保存到CSV文件
        if save_path:
            # 将.txt后缀改为.csv
            csv_path = save_path.replace('.txt', '.csv')
            save_texts_to_csv(text_descriptions, csv_path)
        
        return text_descriptions
    




# import openai
# import time
# import logging
# from typing import List, Dict, Any, Tuple
# from collections import Counter
# import numpy as np
# from scipy import signal
# from fallback_text_generator import FallbackTextGenerator
# from simple_tokenizer import SimpleTokenizer
# from text_utils import save_texts_to_csv, load_texts_from_csv

# class GPTTextGenerator:
#     def __init__(self, args, clip_model=None):
#         """
#         初始化文本生成器  行为全部在文本开头
        
#         Args:
#             args: 配置参数（来自config.py的parse_args()）
#             clip_model: 可选，CLIP模型对象，用于动态token数估算
#         """
#         # 修正getattr参数
#         self.api_key = getattr(args, 'openai_api_key', None)
#         self.model = getattr(args, 'gpt_model', 'gpt-3.5-turbo')
#         self.use_gpt = getattr(args, 'use_gpt', False)
#         self.max_tokens = getattr(args, 'gpt_max_tokens', 70)
#         self.temperature = getattr(args, 'gpt_temperature', 0.5)
#         self.timeout = getattr(args, 'gpt_timeout', 30)
#         self.max_retries = getattr(args, 'gpt_max_retries', 3)
#         self.system_prompt = getattr(args, 'gpt_system_prompt', "")
#         self.logger = logging.getLogger(__name__)
#         self.base_url = getattr(args, 'gpt_base_url', None)
        
#         # 初始化tokenizer用于精确token计数（使用BPE分词）
#         self.tokenizer = SimpleTokenizer(bpe_path="bpe_simple_vocab_16e6.txt.gz")
        
#         # 初始化普通文本生成器
#         self.fallback_generator = FallbackTextGenerator(max_tokens=self.max_tokens)
        
#         # 根据模式进行初始化
#         if self.use_gpt:
#             # GPT模式
#             if self.api_key:
#                 if self.base_url:
#                     self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
#                 else:
#                     self.client = openai.OpenAI(api_key=self.api_key)
#                 self.logger.info(f"Using GPT mode, model: {self.model}, max tokens: {self.max_tokens}, base_url: {self.base_url}")
#             else:
#                 self.logger.warning("GPT mode requires API key, switching to normal mode")
#                 self.use_gpt = False
#         else:
#             # 普通模式
#             self.logger.info(f"Using rule-based text generation mode, max tokens: {self.max_tokens}")
        
#         self.clip_model = clip_model

#         # 特征统计信息存储
#         self.feature_stats = {}
#         self.stats_computed = False

#     def extract_enhanced_features(self, window_data: np.ndarray) -> Dict[str, float]:
#         """
#         直接提取12个区分性特征（专门针对三种行为区分）
        
#         Args:
#             window_data: 传感器数据窗口
            
#         Returns:
#             12个区分性特征字典
#         """
#         if window_data.ndim == 1:
#             window_data = window_data.reshape(-1, 1)
#         # 分离三轴数据
#         x_data = window_data[:, 0]  # 前后方向
#         y_data = window_data[:, 1]  # 左右方向
#         z_data = window_data[:, 2]  # 上下方向
        
#         # 计算合成加速度
#         vm = np.sqrt(x_data**2 + y_data**2 + z_data**2)
        
#         # 1. 周期性特征（反刍的关键特征）
#         # 自相关分析
#         autocorr = np.correlate(vm, vm, mode='full')
#         autocorr = autocorr[len(autocorr)//2:]
#         autocorr_peaks, _ = signal.find_peaks(autocorr[:len(autocorr)//2])
#         autocorr_peak_count = len(autocorr_peaks)
        
#         # 周期性强度
#         if len(autocorr) > 1:
#             periodicity = np.max(autocorr[1:]) / autocorr[0]
#         else:
#             periodicity = 0
        
#         # 2. 头部姿态特征（进食的关键特征）
#         # 俯仰角（前后倾斜）
#         pitch_angles = np.arctan2(x_data, np.sqrt(y_data**2 + z_data**2))
#         pitch_mean = np.mean(pitch_angles)
#         pitch_std = np.std(pitch_angles)
            
#         # 横滚角（左右倾斜）
#         roll_angles = np.arctan2(y_data, np.sqrt(x_data**2 + z_data**2))
#         roll_mean = np.mean(roll_angles)
#         roll_std = np.std(roll_angles)
            
#         # 3. 运动强度特征（其他活动的关键特征）
#         vm_mean = np.mean(vm)
#         vm_std = np.std(vm)
#         energy = np.sum(vm**2)
        
#         # 4. 时序规律性特征
#         peaks, _ = signal.find_peaks(vm)
#         peak_count = len(peaks)
        
#         if len(peaks) > 1:
#             peak_intervals = np.diff(peaks)
#             avg_peak_interval = np.mean(peak_intervals)
#         else:
#             avg_peak_interval = len(vm)
        
#         # 5. 运动复杂度特征
#         # 基于熵的复杂度
#         hist, _ = np.histogram(vm, bins=20)
#         # 归一化为概率分布
#         prob = hist / np.sum(hist)
#         # 只对非零概率计算熵，避免log(0)
#         prob = prob[prob > 0]
#         if len(prob) > 0:
#             complexity = -np.sum(prob * np.log(prob))
#         else:
#             complexity = 0
        
#         # 过零点数
#         zero_crossings = np.sum(np.diff(np.signbit(vm - np.mean(vm))) != 0)
        
#         return {
#             # 周期性特征（反刍关键）
#             'periodicity': periodicity,
#             'autocorr_peak_count': autocorr_peak_count,
            
#             # 头部姿态特征（进食关键）
#             'pitch_mean': pitch_mean,
#             'pitch_std': pitch_std,
#             'roll_mean': roll_mean,
#             'roll_std': roll_std,
            
#             # 运动强度特征（其他活动关键）
#             'vm_mean': vm_mean,
#             'vm_std': vm_std,
#             'energy': energy,
            
#             # 时序规律性特征
#             'peak_count': peak_count,
#             'avg_peak_interval': avg_peak_interval,
            
#             # 运动复杂度特征
#             'complexity': complexity,
#             'zero_crossings': zero_crossings
#         }

#     def compute_feature_statistics(self, all_features_list):
#         """
#         计算特征的统计分布，用于动态阈值

#         Args:
#             all_features_list: 所有窗口的特征字典列表
#         """
#         if not all_features_list:
#             return

#         # 计算关键特征的统计分布
#         key_features = ['periodicity', 'vm_mean', 'complexity', 'pitch_mean', 'energy']

#         for feature_name in key_features:
#             values = [f[feature_name] for f in all_features_list if feature_name in f]
#             if values:
#                 self.feature_stats[feature_name] = {
#                     'mean': np.mean(values),
#                     'std': np.std(values),
#                     'p25': np.percentile(values, 25),
#                     'p50': np.percentile(values, 50),
#                     'p75': np.percentile(values, 75),
#                     'p80': np.percentile(values, 80),
#                     'p85': np.percentile(values, 85),
#                     'p90': np.percentile(values, 90)
#                 }

#         self.stats_computed = True
#         print(f"特征统计计算完成，涵盖 {len(key_features)} 个关键特征")


#     # 标签数字到行为名称的映射
#     LABEL_MAP = {
#         0: "Other",
#         1: "Rumination",
#         2: "Feeding"
#     }



#     def generate_enhanced_prompt(self, features: Dict[str, float], label: int, 
#                                 behavior_type: str, window_data: np.ndarray, 
#                                 window_labels: np.ndarray) -> str:
#         """
#         生成优化的Prompt
        
#         Args:
#             features: 特征字典
#             label: 行为标签
#             behavior_type: 行为类型
#             window_data: 窗口数据
#             window_labels: 窗口标签
            
#         Returns:
#             优化的Prompt字符串
#         """
#         # 直接生成数据摘要，使用所有13个核心特征
#         data_summary = self._generate_data_summary(window_data, features, window_labels)
#         # 统计窗口主标签和分布
#         if window_labels is not None and len(window_labels) > 0:
#             label_counter = Counter(window_labels)
#             main_label_num, main_label_count = label_counter.most_common(1)[0]
#             main_label_name = self.LABEL_MAP.get(main_label_num, str(main_label_num))
#             total = len(window_labels)
#             dist_str = ", ".join([
#                 f"{self.LABEL_MAP.get(lbl, str(lbl))}({cnt/total:.0%})"
#                 for lbl, cnt in label_counter.items()
#             ])
#             label_info = f"Window main label: {main_label_name} ({main_label_num}); Distribution: {dist_str}"
#         else:
#             label_info = "Window main label: Unknown; Distribution: Unknown"
#         # 传递给下一级
#         prompt = self._generate_optimized_prompt(label_info, data_summary, features)
#         return prompt.strip()



#     def _generate_data_summary(self, window_data: np.ndarray, features: Dict[str, float],
#                              window_labels: np.ndarray) -> str:
#         """生成行为区分性摘要"""
#         # 提取关键特征
#         periodicity = features.get('periodicity', 0)
#         pitch_mean = features.get('pitch_mean', 0)
#         pitch_std = features.get('pitch_std', 0)
#         vm_mean = features.get('vm_mean', 0)
#         complexity = features.get('complexity', 0)

#         enhanced_indicators = []

#         # 1. 周期性特征描述（使用动态阈值）
#         if self.stats_computed and 'periodicity' in self.feature_stats:
#             stats = self.feature_stats['periodicity']
#             if periodicity > stats['p90']:
#                 enhanced_indicators.append(f"very strong rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > stats['p80']:
#                 enhanced_indicators.append(f"strong rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > stats['p50']:
#                 enhanced_indicators.append(f"moderate rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > stats['p25']:
#                 enhanced_indicators.append(f"weak rhythmic pattern (periodicity={periodicity:.3f})")
#         else:
#             # 回退到固定阈值
#             if periodicity > 0.6:
#                 enhanced_indicators.append(f"very strong rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > 0.4:
#                 enhanced_indicators.append(f"strong rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > 0.2:
#                 enhanced_indicators.append(f"moderate rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > 0:
#                 enhanced_indicators.append(f"weak rhythmic pattern (periodicity={periodicity:.3f})")

#         # 2. 头部姿态描述
#         if abs(pitch_mean) > 0:
#             pitch_normalized = abs(pitch_mean) / (pitch_std + 1e-6)
#             if pitch_normalized > 2.0:
#                 if pitch_mean < 0:
#                     enhanced_indicators.append(f"significant head down (pitch={pitch_mean:.3f})")
#                 else:
#                     enhanced_indicators.append(f"significant head up (pitch={pitch_mean:.3f})")
#             elif pitch_normalized > 1.0:
#                 if pitch_mean < 0:
#                     enhanced_indicators.append(f"moderate head down (pitch={pitch_mean:.3f})")
#                 else:
#                     enhanced_indicators.append(f"moderate head up (pitch={pitch_mean:.3f})")

#         # 3. 运动强度描述（使用动态阈值，放宽标准）
#         if self.stats_computed and 'vm_mean' in self.feature_stats:
#             stats = self.feature_stats['vm_mean']
#             if vm_mean > stats['p85']:
#                 enhanced_indicators.append(f"high intensity (vm={vm_mean:.3f})")
#             elif vm_mean > stats['p50']:
#                 enhanced_indicators.append(f"moderate intensity (vm={vm_mean:.3f})")
#             elif vm_mean > 0:
#                 enhanced_indicators.append(f"low intensity (vm={vm_mean:.3f})")
#         else:
#             # 回退到固定阈值
#             if vm_mean > 1.5:
#                 enhanced_indicators.append(f"high intensity (vm={vm_mean:.3f})")
#             elif vm_mean > 0.8:
#                 enhanced_indicators.append(f"moderate intensity (vm={vm_mean:.3f})")
#             elif vm_mean > 0:
#                 enhanced_indicators.append(f"low intensity (vm={vm_mean:.3f})")

#         # 4. 复杂度描述（使用动态阈值，放宽标准）
#         if self.stats_computed and 'complexity' in self.feature_stats:
#             stats = self.feature_stats['complexity']
#             if complexity > stats['p85']:
#                 enhanced_indicators.append(f"very complex pattern (complexity={complexity:.3f})")
#             elif complexity > stats['p50']:
#                 enhanced_indicators.append(f"complex pattern (complexity={complexity:.3f})")
#             elif complexity > 0:
#                 enhanced_indicators.append(f"simple pattern (complexity={complexity:.3f})")
#         else:
#             # 回退到固定阈值
#             if complexity > 3.0:
#                 enhanced_indicators.append(f"very complex pattern (complexity={complexity:.3f})")
#             elif complexity > 2.0:
#                 enhanced_indicators.append(f"complex pattern (complexity={complexity:.3f})")
#             elif complexity > 0:
#                 enhanced_indicators.append(f"simple pattern (complexity={complexity:.3f})")

#         # 组合摘要（限制数量）
#         if len(enhanced_indicators) > 4:
#             enhanced_indicators = enhanced_indicators[:4]

#         # 生成最终摘要
#         if enhanced_indicators:
#             summary = ", ".join(enhanced_indicators)
#         else:
#             if vm_mean > 0:
#                 summary = f"detectable movement activity (vm={vm_mean:.3f})"
#             else:
#                 summary = "minimal movement detected"
#         return summary


#     def _generate_optimized_prompt(self, label_info: str, data_summary: str, features: Dict[str, float]) -> str:
#         examples = """Examples of concise behavioral descriptions:
# - FEEDING with head-down posture + moderate intensity → "Feeding behavior with head-down posture"
# - RUMINATION with strong periodicity + low intensity → "Rumination activity with rhythmic jaw movements"
# - OTHER with irregular patterns + high intensity → "Other activity with irregular movement patterns"
# """

#         prompt = f'''
# {examples}

# {label_info}
# Based on the examples above, analyze the following movement features and generate a concise behavioral description.
# The description must clearly include the specific behavior type from the window label.

# Requirements:
# - Must include the exact behavior type name (feeding/rumination/other) in the description
# - Follow the concise style of the examples above
# - For FEEDING: focus on head posture and movement steadiness
# - For RUMINATION: focus on rhythmic periodicity and repetitive motions
# - For OTHER: focus on movement irregularity and intensity variations
# - Use simple, direct behavioral language without commas
# - Avoid complex adjectives and unnecessary details
# - Generate 6-12 words that capture the core behavioral characteristics
# - Do not use commas in the description
# - Use simple phrases without punctuation

# Movement feature summary: {data_summary}

# Behavioral description:'''
#         return prompt




#     def call_gpt_api(self, prompt: str, window_data: np.ndarray, window_labels: np.ndarray, features: Dict[str, float] = None) -> str:
#         """
#         调用GPT API生成文本
#         """
#         max_bpe_tokens = 70
#         for attempt in range(self.max_retries):
#             try:
#                 # 动态调整max_tokens
#                 dynamic_max_tokens = min(self.max_tokens, 70)
#                 if hasattr(self, 'clip_model') and self.clip_model is not None:
#                     try:
#                         import clip
#                         base_tokens = clip.tokenize([prompt])[0]
#                         base_token_count = (base_tokens != 0).sum().item()
#                         dynamic_max_tokens = max(10, 77 - base_token_count - 5)
#                     except Exception as e:
#                         pass
#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": self.system_prompt},
#                         {"role": "user", "content": prompt}
#                     ],
#                     max_tokens=dynamic_max_tokens,
#                     temperature=self.temperature,
#                     timeout=self.timeout
#                 )
#                 generated_text = response.choices[0].message.content.strip()
#                 # 检查BPE分词后token数，超限则重试
#                 bpe_tokens = self.tokenizer.encode(generated_text)
#                 if len(bpe_tokens) <= max_bpe_tokens:
#                     return generated_text
#                 else:
#                     self.logger.warning(f"Generated text BPE tokens {len(bpe_tokens)} > {max_bpe_tokens}, retrying...")
#             except Exception as e:
#                 print(f"API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
#                 if attempt < self.max_retries - 1:
#                     time.sleep(2 ** attempt)  # 指数退避
#                 else:
#                     self.logger.warning(f"GPT API call failed, falling back to fallback generator")
#                     return self.fallback_generator.generate_text_for_window(window_data, window_labels)
#         # 多次重试后仍超限，降级到fallback生成器
#         self.logger.error(f"Failed to generate text within {max_bpe_tokens} BPE tokens after {self.max_retries} attempts. Falling back to fallback generator.")
#         return self.fallback_generator.generate_text_for_window(window_data, window_labels)
    
#     def generate_text_for_window(self, window_data: np.ndarray, window_labels: np.ndarray) -> str:
#         """
#         为单个窗口生成文本描述
        
#         Args:
#             window_data: 传感器数据窗口 [window_size, 3]
#             window_labels: 窗口内所有标签 [window_size]
            
#         Returns:
#             生成的文本描述
#         """
#         if self.use_gpt:
#             # GPT模式：使用GPT API生成文本
#             return self._generate_gpt_text(window_data, window_labels)
#         else:
#             # 普通模式：使用基于规则的文本生成器
#             return self.fallback_generator.generate_text_for_window(window_data, window_labels)
    
#     def _generate_gpt_text(self, window_data: np.ndarray, window_labels: np.ndarray) -> str:
#         """
#         使用GPT API生成文本描述
        
#         Args:
#             window_data: 传感器数据窗口
#             window_labels: 窗口标签
            
#         Returns:
#             GPT生成的文本描述
#         """
#         # 提取增强的窗口特征
#         features = self.extract_enhanced_features(window_data)

#         # 使用统一的Prompt生成
#         prompt = self.generate_enhanced_prompt(features, 0, "", window_data, window_labels)

#         # 调用GPT API，传递features避免重复提取
#         text_description = self.call_gpt_api(prompt, window_data, window_labels, features)
        
#         return text_description

#     def generate_text_for_window_with_features(self, window_data: np.ndarray, window_labels: np.ndarray, cached_features: Dict[str, float]) -> str:
#         """
#         使用缓存的特征为窗口生成文本描述（避免重复特征提取）

#         Args:
#             window_data: 传感器数据窗口
#             window_labels: 窗口标签
#             cached_features: 预先计算的特征

#         Returns:
#             生成的文本描述
#         """
#         if self.use_gpt:
#             # GPT模式：使用缓存的特征
#             prompt = self.generate_enhanced_prompt(cached_features, 0, "", window_data, window_labels)
#             return self.call_gpt_api(prompt, window_data, window_labels, cached_features)
#         else:
#             # 普通模式：使用基于规则的文本生成器
#             return self.fallback_generator.generate_text_for_window(window_data, window_labels)

#     def generate_text_for_window(self, window_data: np.ndarray, window_labels: np.ndarray) -> str:
#         """
#         为单个窗口生成文本描述

#         Args:
#             window_data: 传感器数据窗口 [window_size, 3]
#             window_labels: 窗口内所有标签 [window_size]

#         Returns:
#             生成的文本描述
#         """
#         if self.use_gpt:
#             # GPT模式：使用GPT API生成文本
#             return self._generate_gpt_text(window_data, window_labels)
#         else:
#             # 普通模式：使用基于规则的文本生成器
#             return self.fallback_generator.generate_text_for_window(window_data, window_labels)

#     def generate_texts_for_windows_batch(self, features: np.ndarray, labels: np.ndarray,
#                                        window_size: int, stride: int, batch_size: int = 32,
#                                        save_path: str = None) -> List[str]:
#         """
#         分批为所有窗口生成文本描述（内存优化版本）
        
#         Args:
#             features: 传感器数据
#             labels: 标签数据
#             window_size: 窗口大小
#             stride: 窗口步长
#             batch_size: 批处理大小
#             save_path: 保存路径（可选）
            
#         Returns:
#             文本描述列表
#         """
#         text_descriptions = []

#         if self.use_gpt:
#             print(f"Starting batch generation of window text descriptions using {self.model}...")
#             print("Mode: GPT API generation (batch processing)")
#         else:
#             print("Starting batch generation of window text descriptions using rule-based text generator...")
#             print("Mode: Rule-based generation (batch processing)")
#         print(f"Window size: {window_size}, Stride: {stride}, Batch size: {batch_size}")

#         # 计算窗口数量
#         num_windows = (len(features) - window_size) // stride + 1
#         print(f"Total windows: {num_windows}")

#         # 预先计算所有窗口的特征并缓存（避免重复计算）
#         print("提取所有窗口特征...")
#         all_features = {}  # 缓存所有窗口的特征
#         for window_idx in range(num_windows):
#             start_idx = window_idx * stride
#             end_idx = start_idx + window_size
#             window_data = features[start_idx:end_idx]
#             window_features = self.extract_enhanced_features(window_data)
#             all_features[window_idx] = window_features

#         # 计算特征统计分布（用于动态阈值）
#         if not self.stats_computed:
#             print("计算特征统计分布...")
#             self.compute_feature_statistics(list(all_features.values()))
#             print("特征统计计算完成")
        
#         # 分批处理
#         for batch_start in range(0, num_windows, batch_size):
#             batch_end = min(batch_start + batch_size, num_windows)
#             batch_texts = []
            
#             print(f"Processing text batch {batch_start//batch_size + 1}/{(num_windows + batch_size - 1)//batch_size}")
            
#             for window_idx in range(batch_start, batch_end):
#                 start_idx = window_idx * stride
#                 end_idx = start_idx + window_size

#                 # 获取窗口数据
#                 window_data = features[start_idx:end_idx]
#                 window_labels = labels[start_idx:end_idx]

#                 # 使用缓存的特征生成文本描述
#                 cached_features = all_features[window_idx]
#                 text = self.generate_text_for_window_with_features(window_data, window_labels, cached_features)
#                 batch_texts.append(text)
                
#                 # 只在GPT模式下添加延迟（避免API限制）
#                 if self.use_gpt:
#                     time.sleep(0.1)
            
#             # 添加到总列表
#             text_descriptions.extend(batch_texts)
            
#             # 清理内存
#             del batch_texts
#             import gc
#             gc.collect()
        
#         print(f"Window text generation completed, generated {len(text_descriptions)} descriptions")
        
#         # 保存到CSV文件
#         if save_path:
#             # 将.txt后缀改为.csv
#             csv_path = save_path.replace('.txt', '.csv')
#             save_texts_to_csv(text_descriptions, csv_path)
        
#         return text_descriptions
    






# import openai
# import time
# import logging
# from typing import List, Dict, Any, Tuple
# from collections import Counter
# import numpy as np
# from scipy import signal
# from fallback_text_generator import FallbackTextGenerator
# from simple_tokenizer import SimpleTokenizer
# from text_utils import save_texts_to_csv, load_texts_from_csv

# class GPTTextGenerator:
#     def __init__(self, args, clip_model=None):
#         """
#         初始化文本生成器  行为全部在文本开头
        
#         Args:
#             args: 配置参数（来自config.py的parse_args()）
#             clip_model: 可选，CLIP模型对象，用于动态token数估算
#         """
#         # 修正getattr参数
#         self.api_key = getattr(args, 'openai_api_key', None)
#         self.model = getattr(args, 'gpt_model', 'gpt-3.5-turbo')
#         self.use_gpt = getattr(args, 'use_gpt', False)
#         self.max_tokens = getattr(args, 'gpt_max_tokens', 70)
#         self.temperature = getattr(args, 'gpt_temperature', 0.5)
#         self.timeout = getattr(args, 'gpt_timeout', 30)
#         self.max_retries = getattr(args, 'gpt_max_retries', 3)
#         self.system_prompt = getattr(args, 'gpt_system_prompt', "")
#         self.logger = logging.getLogger(__name__)
#         self.base_url = getattr(args, 'gpt_base_url', None)
        
#         # 初始化tokenizer用于精确token计数（使用BPE分词）
#         self.tokenizer = SimpleTokenizer(bpe_path="bpe_simple_vocab_16e6.txt.gz")
        
#         # 初始化普通文本生成器
#         self.fallback_generator = FallbackTextGenerator(max_tokens=self.max_tokens)
        
#         # 根据模式进行初始化
#         if self.use_gpt:
#             # GPT模式
#             if self.api_key:
#                 if self.base_url:
#                     self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
#                 else:
#                     self.client = openai.OpenAI(api_key=self.api_key)
#                 self.logger.info(f"Using GPT mode, model: {self.model}, max tokens: {self.max_tokens}, base_url: {self.base_url}")
#             else:
#                 self.logger.warning("GPT mode requires API key, switching to normal mode")
#                 self.use_gpt = False
#         else:
#             # 普通模式
#             self.logger.info(f"Using rule-based text generation mode, max tokens: {self.max_tokens}")
        
#         self.clip_model = clip_model

#         # 特征统计信息存储
#         self.feature_stats = {}
#         self.stats_computed = False

#     def extract_enhanced_features(self, window_data: np.ndarray) -> Dict[str, float]:
#         """
#         直接提取12个区分性特征（专门针对三种行为区分）
        
#         Args:
#             window_data: 传感器数据窗口
            
#         Returns:
#             12个区分性特征字典
#         """
#         if window_data.ndim == 1:
#             window_data = window_data.reshape(-1, 1)
#         # 分离三轴数据
#         x_data = window_data[:, 0]  # 前后方向
#         y_data = window_data[:, 1]  # 左右方向
#         z_data = window_data[:, 2]  # 上下方向
        
#         # 计算合成加速度
#         vm = np.sqrt(x_data**2 + y_data**2 + z_data**2)
        
#         # 1. 周期性特征（反刍的关键特征）
#         # 自相关分析
#         autocorr = np.correlate(vm, vm, mode='full')
#         autocorr = autocorr[len(autocorr)//2:]
#         autocorr_peaks, _ = signal.find_peaks(autocorr[:len(autocorr)//2])
#         autocorr_peak_count = len(autocorr_peaks)
        
#         # 周期性强度
#         if len(autocorr) > 1:
#             periodicity = np.max(autocorr[1:]) / autocorr[0]
#         else:
#             periodicity = 0
        
#         # 2. 头部姿态特征（进食的关键特征）
#         # 俯仰角（前后倾斜）
#         pitch_angles = np.arctan2(x_data, np.sqrt(y_data**2 + z_data**2))
#         pitch_mean = np.mean(pitch_angles)
#         pitch_std = np.std(pitch_angles)
            
#         # 横滚角（左右倾斜）
#         roll_angles = np.arctan2(y_data, np.sqrt(x_data**2 + z_data**2))
#         roll_mean = np.mean(roll_angles)
#         roll_std = np.std(roll_angles)
            
#         # 3. 运动强度特征（其他活动的关键特征）
#         vm_mean = np.mean(vm)
#         vm_std = np.std(vm)
#         energy = np.sum(vm**2)
        
#         # 4. 时序规律性特征
#         peaks, _ = signal.find_peaks(vm)
#         peak_count = len(peaks)
        
#         if len(peaks) > 1:
#             peak_intervals = np.diff(peaks)
#             avg_peak_interval = np.mean(peak_intervals)
#         else:
#             avg_peak_interval = len(vm)
        
#         # 5. 运动复杂度特征
#         # 基于熵的复杂度
#         # hist, _ = np.histogram(vm, bins=20)
#         # complexity = -np.sum(hist * np.log(hist + 1e-10))

#         # 基于熵的复杂度
#         hist, _ = np.histogram(vm, bins=20)
#         # 归一化为概率分布
#         prob = hist / np.sum(hist)
#         # 只对非零概率计算熵，避免log(0)
#         prob = prob[prob > 0]
#         if len(prob) > 0:
#             complexity = -np.sum(prob * np.log(prob))
#         else:
#             complexity = 0
        
#         # 过零点数
#         zero_crossings = np.sum(np.diff(np.signbit(vm - np.mean(vm))) != 0)
        
#         return {
#             # 周期性特征（反刍关键）
#             'periodicity': periodicity,
#             'autocorr_peak_count': autocorr_peak_count,
            
#             # 头部姿态特征（进食关键）
#             'pitch_mean': pitch_mean,
#             'pitch_std': pitch_std,
#             'roll_mean': roll_mean,
#             'roll_std': roll_std,
            
#             # 运动强度特征（其他活动关键）
#             'vm_mean': vm_mean,
#             'vm_std': vm_std,
#             'energy': energy,
            
#             # 时序规律性特征
#             'peak_count': peak_count,
#             'avg_peak_interval': avg_peak_interval,
            
#             # 运动复杂度特征
#             'complexity': complexity,
#             'zero_crossings': zero_crossings
#         }

#     def compute_feature_statistics(self, all_features_list):
#         """
#         计算特征的统计分布，用于动态阈值

#         Args:
#             all_features_list: 所有窗口的特征字典列表
#         """
#         if not all_features_list:
#             return

#         # 计算关键特征的统计分布
#         key_features = ['periodicity', 'vm_mean', 'complexity', 'pitch_mean', 'energy']

#         for feature_name in key_features:
#             values = [f[feature_name] for f in all_features_list if feature_name in f]
#             if values:
#                 self.feature_stats[feature_name] = {
#                     'mean': np.mean(values),
#                     'std': np.std(values),
#                     'p25': np.percentile(values, 25),
#                     'p50': np.percentile(values, 50),
#                     'p75': np.percentile(values, 75),
#                     'p80': np.percentile(values, 80),
#                     'p85': np.percentile(values, 85),
#                     'p90': np.percentile(values, 90)
#                 }

#         self.stats_computed = True
#         print(f"特征统计计算完成，涵盖 {len(key_features)} 个关键特征")


#     # 标签数字到行为名称的映射
#     LABEL_MAP = {
#         0: "Other",
#         1: "Rumination",
#         2: "Feeding"
#     }



#     def generate_enhanced_prompt(self, features: Dict[str, float], label: int, 
#                                 behavior_type: str, window_data: np.ndarray, 
#                                 window_labels: np.ndarray) -> str:
#         """
#         生成优化的Prompt
        
#         Args:
#             features: 特征字典
#             label: 行为标签
#             behavior_type: 行为类型
#             window_data: 窗口数据
#             window_labels: 窗口标签
            
#         Returns:
#             优化的Prompt字符串
#         """
#         # 直接生成数据摘要，使用所有13个核心特征
#         data_summary = self._generate_data_summary(window_data, features, window_labels)
#         # 统计窗口主标签和分布
#         if window_labels is not None and len(window_labels) > 0:
#             label_counter = Counter(window_labels)
#             main_label_num, main_label_count = label_counter.most_common(1)[0]
#             main_label_name = self.LABEL_MAP.get(main_label_num, str(main_label_num))
#             total = len(window_labels)
#             dist_str = ", ".join([
#                 f"{self.LABEL_MAP.get(lbl, str(lbl))}({cnt/total:.0%})"
#                 for lbl, cnt in label_counter.items()
#             ])
#             label_info = f"Window main label: {main_label_name} ({main_label_num}); Distribution: {dist_str}"
#         else:
#             label_info = "Window main label: Unknown; Distribution: Unknown"
#         # 传递给下一级
#         prompt = self._generate_optimized_prompt(label_info, data_summary, features)
#         return prompt.strip()



#     def _generate_data_summary(self, window_data: np.ndarray, features: Dict[str, float],
#                              window_labels: np.ndarray) -> str:
#         """生成行为区分性摘要"""
#         # 提取关键特征
#         periodicity = features.get('periodicity', 0)
#         pitch_mean = features.get('pitch_mean', 0)
#         pitch_std = features.get('pitch_std', 0)
#         vm_mean = features.get('vm_mean', 0)
#         complexity = features.get('complexity', 0)

#         enhanced_indicators = []

#         # 1. 周期性特征描述（使用动态阈值）
#         if self.stats_computed and 'periodicity' in self.feature_stats:
#             stats = self.feature_stats['periodicity']
#             if periodicity > stats['p90']:
#                 enhanced_indicators.append(f"very strong rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > stats['p80']:
#                 enhanced_indicators.append(f"strong rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > stats['p50']:
#                 enhanced_indicators.append(f"moderate rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > stats['p25']:
#                 enhanced_indicators.append(f"weak rhythmic pattern (periodicity={periodicity:.3f})")
#         else:
#             # 回退到固定阈值
#             if periodicity > 0.6:
#                 enhanced_indicators.append(f"very strong rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > 0.4:
#                 enhanced_indicators.append(f"strong rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > 0.2:
#                 enhanced_indicators.append(f"moderate rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > 0:
#                 enhanced_indicators.append(f"weak rhythmic pattern (periodicity={periodicity:.3f})")

#         # 2. 头部姿态描述
#         if abs(pitch_mean) > 0:
#             pitch_normalized = abs(pitch_mean) / (pitch_std + 1e-6)
#             if pitch_normalized > 2.0:
#                 if pitch_mean < 0:
#                     enhanced_indicators.append(f"significant head down (pitch={pitch_mean:.3f})")
#                 else:
#                     enhanced_indicators.append(f"significant head up (pitch={pitch_mean:.3f})")
#             elif pitch_normalized > 1.0:
#                 if pitch_mean < 0:
#                     enhanced_indicators.append(f"moderate head down (pitch={pitch_mean:.3f})")
#                 else:
#                     enhanced_indicators.append(f"moderate head up (pitch={pitch_mean:.3f})")

#         # 3. 运动强度描述（使用动态阈值，放宽标准）
#         if self.stats_computed and 'vm_mean' in self.feature_stats:
#             stats = self.feature_stats['vm_mean']
#             if vm_mean > stats['p85']:
#                 enhanced_indicators.append(f"high intensity (vm={vm_mean:.3f})")
#             elif vm_mean > stats['p50']:
#                 enhanced_indicators.append(f"moderate intensity (vm={vm_mean:.3f})")
#             elif vm_mean > 0:
#                 enhanced_indicators.append(f"low intensity (vm={vm_mean:.3f})")
#         else:
#             # 回退到固定阈值
#             if vm_mean > 1.5:
#                 enhanced_indicators.append(f"high intensity (vm={vm_mean:.3f})")
#             elif vm_mean > 0.8:
#                 enhanced_indicators.append(f"moderate intensity (vm={vm_mean:.3f})")
#             elif vm_mean > 0:
#                 enhanced_indicators.append(f"low intensity (vm={vm_mean:.3f})")

#         # 4. 复杂度描述（使用动态阈值，放宽标准）
#         if self.stats_computed and 'complexity' in self.feature_stats:
#             stats = self.feature_stats['complexity']
#             if complexity > stats['p85']:
#                 enhanced_indicators.append(f"very complex pattern (complexity={complexity:.3f})")
#             elif complexity > stats['p50']:
#                 enhanced_indicators.append(f"complex pattern (complexity={complexity:.3f})")
#             elif complexity > 0:
#                 enhanced_indicators.append(f"simple pattern (complexity={complexity:.3f})")
#         else:
#             # 回退到固定阈值
#             if complexity > 3.0:
#                 enhanced_indicators.append(f"very complex pattern (complexity={complexity:.3f})")
#             elif complexity > 2.0:
#                 enhanced_indicators.append(f"complex pattern (complexity={complexity:.3f})")
#             elif complexity > 0:
#                 enhanced_indicators.append(f"simple pattern (complexity={complexity:.3f})")

#         # 组合摘要（限制数量）
#         if len(enhanced_indicators) > 4:
#             enhanced_indicators = enhanced_indicators[:4]

#         # 生成最终摘要
#         if enhanced_indicators:
#             summary = ", ".join(enhanced_indicators)
#         else:
#             if vm_mean > 0:
#                 summary = f"detectable movement activity (vm={vm_mean:.3f})"
#             else:
#                 summary = "minimal movement detected"
#         return summary


#     def _generate_optimized_prompt(self, label_info: str, data_summary: str, features: Dict[str, float]) -> str:
#         examples = """Examples of concise behavioral descriptions:
# - FEEDING with head-down posture + moderate intensity → "Feeding behavior with head-down posture"
# - RUMINATION with strong periodicity + low intensity → "Rumination activity with rhythmic jaw movements"
# - OTHER with irregular patterns + high intensity → "Other activity with irregular movement patterns"
# """

#         prompt = f'''
# {examples}

# {label_info}
# Based on the examples above, analyze the following movement features and generate a concise behavioral description.
# The description must clearly include the specific behavior type from the window label.

# Requirements:
# - Must include the exact behavior type name (feeding/rumination/other) in the description
# - Follow the concise style of the examples above
# - For FEEDING: focus on head posture and movement steadiness
# - For RUMINATION: focus on rhythmic periodicity and repetitive motions
# - For OTHER: focus on movement irregularity and intensity variations
# - Use simple, direct behavioral language without commas
# - Avoid complex adjectives and unnecessary details
# - Generate 6-12 words that capture the core behavioral characteristics
# - Do not use commas in the description
# - Use simple phrases without punctuation

# Movement feature summary: {data_summary}

# Behavioral description:'''
#         return prompt




#     def call_gpt_api(self, prompt: str, window_data: np.ndarray, window_labels: np.ndarray, features: Dict[str, float] = None) -> str:
#         """
#         调用GPT API生成文本
#         """
#         max_bpe_tokens = 70
#         for attempt in range(self.max_retries):
#             try:
#                 # 动态调整max_tokens
#                 dynamic_max_tokens = min(self.max_tokens, 70)
#                 if hasattr(self, 'clip_model') and self.clip_model is not None:
#                     try:
#                         import clip
#                         base_tokens = clip.tokenize([prompt])[0]
#                         base_token_count = (base_tokens != 0).sum().item()
#                         dynamic_max_tokens = max(10, 77 - base_token_count - 5)
#                     except Exception as e:
#                         pass
#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": self.system_prompt},
#                         {"role": "user", "content": prompt}
#                     ],
#                     max_tokens=dynamic_max_tokens,
#                     temperature=self.temperature,
#                     timeout=self.timeout
#                 )
#                 generated_text = response.choices[0].message.content.strip()
#                 # 检查BPE分词后token数，超限则重试
#                 bpe_tokens = self.tokenizer.encode(generated_text)
#                 if len(bpe_tokens) <= max_bpe_tokens:
#                     return generated_text
#                 else:
#                     self.logger.warning(f"Generated text BPE tokens {len(bpe_tokens)} > {max_bpe_tokens}, retrying...")
#             except Exception as e:
#                 print(f"API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
#                 if attempt < self.max_retries - 1:
#                     time.sleep(2 ** attempt)  # 指数退避
#                 else:
#                     self.logger.warning(f"GPT API call failed, falling back to fallback generator")
#                     return self.fallback_generator.generate_text_for_window(window_data, window_labels)
#         # 多次重试后仍超限，降级到fallback生成器
#         self.logger.error(f"Failed to generate text within {max_bpe_tokens} BPE tokens after {self.max_retries} attempts. Falling back to fallback generator.")
#         return self.fallback_generator.generate_text_for_window(window_data, window_labels)
    
#     def generate_text_for_window(self, window_data: np.ndarray, window_labels: np.ndarray) -> str:
#         """
#         为单个窗口生成文本描述
        
#         Args:
#             window_data: 传感器数据窗口 [window_size, 3]
#             window_labels: 窗口内所有标签 [window_size]
            
#         Returns:
#             生成的文本描述
#         """
#         if self.use_gpt:
#             # GPT模式：使用GPT API生成文本
#             return self._generate_gpt_text(window_data, window_labels)
#         else:
#             # 普通模式：使用基于规则的文本生成器
#             return self.fallback_generator.generate_text_for_window(window_data, window_labels)
    
#     def _generate_gpt_text(self, window_data: np.ndarray, window_labels: np.ndarray) -> str:
#         """
#         使用GPT API生成文本描述
        
#         Args:
#             window_data: 传感器数据窗口
#             window_labels: 窗口标签
            
#         Returns:
#             GPT生成的文本描述
#         """
#         # 提取增强的窗口特征
#         features = self.extract_enhanced_features(window_data)

#         # 使用统一的Prompt生成
#         prompt = self.generate_enhanced_prompt(features, 0, "", window_data, window_labels)

#         # 调用GPT API，传递features避免重复提取
#         text_description = self.call_gpt_api(prompt, window_data, window_labels, features)
        
#         return text_description

#     def generate_text_for_window_with_features(self, window_data: np.ndarray, window_labels: np.ndarray, cached_features: Dict[str, float]) -> str:
#         """
#         使用缓存的特征为窗口生成文本描述（避免重复特征提取）

#         Args:
#             window_data: 传感器数据窗口
#             window_labels: 窗口标签
#             cached_features: 预先计算的特征

#         Returns:
#             生成的文本描述
#         """
#         if self.use_gpt:
#             # GPT模式：使用缓存的特征
#             prompt = self.generate_enhanced_prompt(cached_features, 0, "", window_data, window_labels)
#             return self.call_gpt_api(prompt, window_data, window_labels, cached_features)
#         else:
#             # 普通模式：使用基于规则的文本生成器
#             return self.fallback_generator.generate_text_for_window(window_data, window_labels)

#     def generate_text_for_window(self, window_data: np.ndarray, window_labels: np.ndarray) -> str:
#         """
#         为单个窗口生成文本描述

#         Args:
#             window_data: 传感器数据窗口 [window_size, 3]
#             window_labels: 窗口内所有标签 [window_size]

#         Returns:
#             生成的文本描述
#         """
#         if self.use_gpt:
#             # GPT模式：使用GPT API生成文本
#             return self._generate_gpt_text(window_data, window_labels)
#         else:
#             # 普通模式：使用基于规则的文本生成器
#             return self.fallback_generator.generate_text_for_window(window_data, window_labels)

#     def generate_texts_for_windows_batch(self, features: np.ndarray, labels: np.ndarray,
#                                        window_size: int, stride: int, batch_size: int = 32,
#                                        save_path: str = None) -> List[str]:
#         """
#         分批为所有窗口生成文本描述（内存优化版本）
        
#         Args:
#             features: 传感器数据
#             labels: 标签数据
#             window_size: 窗口大小
#             stride: 窗口步长
#             batch_size: 批处理大小
#             save_path: 保存路径（可选）
            
#         Returns:
#             文本描述列表
#         """
#         text_descriptions = []

#         if self.use_gpt:
#             print(f"Starting batch generation of window text descriptions using {self.model}...")
#             print("Mode: GPT API generation (batch processing)")
#         else:
#             print("Starting batch generation of window text descriptions using rule-based text generator...")
#             print("Mode: Rule-based generation (batch processing)")
#         print(f"Window size: {window_size}, Stride: {stride}, Batch size: {batch_size}")

#         # 计算窗口数量
#         num_windows = (len(features) - window_size) // stride + 1
#         print(f"Total windows: {num_windows}")

#         # 预先计算所有窗口的特征并缓存（避免重复计算）
#         print("提取所有窗口特征...")
#         all_features = {}  # 缓存所有窗口的特征
#         for window_idx in range(num_windows):
#             start_idx = window_idx * stride
#             end_idx = start_idx + window_size
#             window_data = features[start_idx:end_idx]
#             window_features = self.extract_enhanced_features(window_data)
#             all_features[window_idx] = window_features

#         # 计算特征统计分布（用于动态阈值）
#         if not self.stats_computed:
#             print("计算特征统计分布...")
#             self.compute_feature_statistics(list(all_features.values()))
#             print("特征统计计算完成")
        
#         # 分批处理
#         for batch_start in range(0, num_windows, batch_size):
#             batch_end = min(batch_start + batch_size, num_windows)
#             batch_texts = []
            
#             print(f"Processing text batch {batch_start//batch_size + 1}/{(num_windows + batch_size - 1)//batch_size}")
            
#             for window_idx in range(batch_start, batch_end):
#                 start_idx = window_idx * stride
#                 end_idx = start_idx + window_size

#                 # 获取窗口数据
#                 window_data = features[start_idx:end_idx]
#                 window_labels = labels[start_idx:end_idx]

#                 # 使用缓存的特征生成文本描述
#                 cached_features = all_features[window_idx]
#                 text = self.generate_text_for_window_with_features(window_data, window_labels, cached_features)
#                 batch_texts.append(text)
                
#                 # 只在GPT模式下添加延迟（避免API限制）
#                 if self.use_gpt:
#                     time.sleep(0.1)
            
#             # 添加到总列表
#             text_descriptions.extend(batch_texts)
            
#             # 清理内存
#             del batch_texts
#             import gc
#             gc.collect()
        
#         print(f"Window text generation completed, generated {len(text_descriptions)} descriptions")
        
#         # 保存到CSV文件
#         if save_path:
#             # 将.txt后缀改为.csv
#             csv_path = save_path.replace('.txt', '.csv')
#             save_texts_to_csv(text_descriptions, csv_path)
        
#         return text_descriptions
    








# # 固定阈值
# import openai
# import time
# import logging
# from typing import List, Dict, Any, Tuple
# from collections import Counter
# import numpy as np
# from scipy import signal
# from fallback_text_generator import FallbackTextGenerator
# from simple_tokenizer import SimpleTokenizer
# from text_utils import save_texts_to_csv, load_texts_from_csv

# class GPTTextGenerator:
#     def __init__(self, args, clip_model=None):
#         """
#         初始化文本生成器
        
#         Args:
#             args: 配置参数（来自config.py的parse_args()）
#             clip_model: 可选，CLIP模型对象，用于动态token数估算
#         """
#         # 修正getattr参数
#         self.api_key = getattr(args, 'openai_api_key', None)
#         self.model = getattr(args, 'gpt_model', 'gpt-3.5-turbo')
#         self.use_gpt = getattr(args, 'use_gpt', False)
#         self.max_tokens = getattr(args, 'gpt_max_tokens', 70)
#         self.temperature = getattr(args, 'gpt_temperature', 0.5)
#         self.timeout = getattr(args, 'gpt_timeout', 30)
#         self.max_retries = getattr(args, 'gpt_max_retries', 3)
#         self.system_prompt = getattr(args, 'gpt_system_prompt', "")
#         self.logger = logging.getLogger(__name__)
#         self.base_url = getattr(args, 'gpt_base_url', None)
        
#         # 初始化tokenizer用于精确token计数（使用BPE分词）
#         self.tokenizer = SimpleTokenizer(bpe_path="bpe_simple_vocab_16e6.txt.gz")
        
#         # 初始化普通文本生成器
#         self.fallback_generator = FallbackTextGenerator(max_tokens=self.max_tokens)
        
#         # 根据模式进行初始化
#         if self.use_gpt:
#             # GPT模式
#             if self.api_key:
#                 if self.base_url:
#                     self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
#                 else:
#                     self.client = openai.OpenAI(api_key=self.api_key)
#                 self.logger.info(f"Using GPT mode, model: {self.model}, max tokens: {self.max_tokens}, base_url: {self.base_url}")
#             else:
#                 self.logger.warning("GPT mode requires API key, switching to normal mode")
#                 self.use_gpt = False
#         else:
#             # 普通模式
#             self.logger.info(f"Using rule-based text generation mode, max tokens: {self.max_tokens}")
        
#         self.clip_model = clip_model

#         # 特征统计信息存储
#         self.feature_stats = {}
#         self.stats_computed = False

#     def extract_enhanced_features(self, window_data: np.ndarray) -> Dict[str, float]:
#         """
#         直接提取12个区分性特征（专门针对三种行为区分）
        
#         Args:
#             window_data: 传感器数据窗口
            
#         Returns:
#             12个区分性特征字典
#         """
#         if window_data.ndim == 1:
#             window_data = window_data.reshape(-1, 1)
#         # 分离三轴数据
#         x_data = window_data[:, 0]  # 前后方向
#         y_data = window_data[:, 1]  # 左右方向
#         z_data = window_data[:, 2]  # 上下方向
        
#         # 计算合成加速度
#         vm = np.sqrt(x_data**2 + y_data**2 + z_data**2)
        
#         # 1. 周期性特征（反刍的关键特征）
#         # 自相关分析
#         autocorr = np.correlate(vm, vm, mode='full')
#         autocorr = autocorr[len(autocorr)//2:]
#         autocorr_peaks, _ = signal.find_peaks(autocorr[:len(autocorr)//2])
#         autocorr_peak_count = len(autocorr_peaks)
        
#         # 周期性强度
#         if len(autocorr) > 1:
#             periodicity = np.max(autocorr[1:]) / autocorr[0]
#         else:
#             periodicity = 0
        
#         # 2. 头部姿态特征（进食的关键特征）
#         # 俯仰角（前后倾斜）
#         pitch_angles = np.arctan2(x_data, np.sqrt(y_data**2 + z_data**2))
#         pitch_mean = np.mean(pitch_angles)
#         pitch_std = np.std(pitch_angles)
            
#         # 横滚角（左右倾斜）
#         roll_angles = np.arctan2(y_data, np.sqrt(x_data**2 + z_data**2))
#         roll_mean = np.mean(roll_angles)
#         roll_std = np.std(roll_angles)
            
#         # 3. 运动强度特征（其他活动的关键特征）
#         vm_mean = np.mean(vm)
#         vm_std = np.std(vm)
#         energy = np.sum(vm**2)
        
#         # 4. 时序规律性特征
#         peaks, _ = signal.find_peaks(vm)
#         peak_count = len(peaks)
        
#         if len(peaks) > 1:
#             peak_intervals = np.diff(peaks)
#             avg_peak_interval = np.mean(peak_intervals)
#         else:
#             avg_peak_interval = len(vm)
        
#         # 5. 运动复杂度特征
#         # 基于熵的复杂度
#         hist, _ = np.histogram(vm, bins=20)
#         # 归一化为概率分布
#         prob = hist / np.sum(hist)
#         # 只对非零概率计算熵，避免log(0)
#         prob = prob[prob > 0]
#         if len(prob) > 0:
#             complexity = -np.sum(prob * np.log(prob))
#         else:
#             complexity = 0
        
#         # 过零点数
#         zero_crossings = np.sum(np.diff(np.signbit(vm - np.mean(vm))) != 0)
        
#         return {
#             # 周期性特征（反刍关键）
#             'periodicity': periodicity,
#             'autocorr_peak_count': autocorr_peak_count,
            
#             # 头部姿态特征（进食关键）
#             'pitch_mean': pitch_mean,
#             'pitch_std': pitch_std,
#             'roll_mean': roll_mean,
#             'roll_std': roll_std,
            
#             # 运动强度特征（其他活动关键）
#             'vm_mean': vm_mean,
#             'vm_std': vm_std,
#             'energy': energy,
            
#             # 时序规律性特征
#             'peak_count': peak_count,
#             'avg_peak_interval': avg_peak_interval,
            
#             # 运动复杂度特征
#             'complexity': complexity,
#             'zero_crossings': zero_crossings
#         }

#     def compute_feature_statistics(self, all_features_list):
#         """
#         计算特征的统计分布，用于动态阈值

#         Args:
#             all_features_list: 所有窗口的特征字典列表
#         """
#         if not all_features_list:
#             return

#         # 计算关键特征的统计分布
#         key_features = ['periodicity', 'vm_mean', 'complexity', 'pitch_mean', 'energy']

#         for feature_name in key_features:
#             values = [f[feature_name] for f in all_features_list if feature_name in f]
#             if values:
#                 self.feature_stats[feature_name] = {
#                     'mean': np.mean(values),
#                     'std': np.std(values),
#                     'p25': np.percentile(values, 25),
#                     'p50': np.percentile(values, 50),
#                     'p75': np.percentile(values, 75),
#                     'p80': np.percentile(values, 80),
#                     'p85': np.percentile(values, 85),
#                     'p90': np.percentile(values, 90)
#                 }

#         self.stats_computed = True
#         print(f"特征统计计算完成，涵盖 {len(key_features)} 个关键特征")

#     def select_important_features(self, features: Dict[str, float], top_n: int = 6) -> List[Tuple[str, str, float]]:
#         """
#         选择最重要的特征（基于固定阈值）
        
#         Args:
#             features: 特征字典
#             top_n: 选择的特征数量
            
#         Returns:
#             重要特征列表 [(特征名, 描述, 重要性分数)]
#         """
#         scored_features = []
        
#         # 定义固定阈值和重要性权重
#         thresholds = {
#             'periodicity': {'very_high': 0.8, 'high': 0.6, 'medium': 0.3, 'low': 0.0},
#             'vm_mean': {'very_high': 2.5, 'high': 1.5, 'medium': 0.8, 'low': 0.0},
#             'vm_std': {'very_high': 1.5, 'high': 0.8, 'medium': 0.3, 'low': 0.0},
#             'complexity': {'very_high': 3.0, 'high': 2.0, 'medium': 1.0, 'low': 0.0},
#             'energy': {'very_high': 1500, 'high': 800, 'medium': 300, 'low': 0},
#             'pitch_mean': {'high': 20, 'medium': 10, 'low': 0},
#             'roll_mean': {'high': 15, 'medium': 5, 'low': 0},
#             'peak_count': {'high': 7, 'medium': 3, 'low': 0},
#             'autocorr_peak_count': {'high': 8, 'medium': 5, 'low': 2},
#             'zero_crossings': {'high': 15, 'medium': 8, 'low': 3}
#         }
        
#         for feature_name, value in features.items():
#             if feature_name in thresholds:
#                 thresholds_feature = thresholds[feature_name]
                
#                 # 确定特征水平
#                 if 'very_high' in thresholds_feature:
#                     if value >= thresholds_feature['very_high']:
#                         level = 'very_high'
#                     elif value >= thresholds_feature['high']:
#                         level = 'high'
#                     elif value >= thresholds_feature['medium']:
#                         level = 'medium'
#                     else:
#                         level = 'low'
#                 else:
#                     if value >= thresholds_feature['high']:
#                         level = 'high'
#                     elif value >= thresholds_feature['medium']:
#                         level = 'medium'
#                     else:
#                         level = 'low'
                
#                 # 生成描述
#                 if feature_name == 'periodicity':
#                     if level == 'very_high':
#                         description = f"very strong rhythmic pattern (periodicity={value:.3f})"
#                     elif level == 'high':
#                         description = f"strong rhythmic pattern (periodicity={value:.3f})"
#                     elif level == 'medium':
#                         description = f"moderate rhythmic pattern (periodicity={value:.3f})"
#                     else:
#                         description = f"weak rhythmic pattern (periodicity={value:.3f})"
#                 elif feature_name == 'vm_mean':
#                     if level == 'very_high':
#                         description = f"very high intensity (vm={value:.3f})"
#                     elif level == 'high':
#                         description = f"high intensity (vm={value:.3f})"
#                     elif level == 'medium':
#                         description = f"moderate intensity (vm={value:.3f})"
#                     else:
#                         description = f"low intensity (vm={value:.3f})"
#                 elif feature_name == 'vm_std':
#                     if level == 'very_high':
#                         description = f"highly irregular movement (vm_std={value:.3f})"
#                     elif level == 'high':
#                         description = f"irregular movement (vm_std={value:.3f})"
#                     elif level == 'medium':
#                         description = f"moderately regular movement (vm_std={value:.3f})"
#                     else:
#                         description = f"regular movement (vm_std={value:.3f})"
#                 elif feature_name == 'complexity':
#                     if level == 'very_high':
#                         description = f"very complex pattern (complexity={value:.3f})"
#                     elif level == 'high':
#                         description = f"complex pattern (complexity={value:.3f})"
#                     elif level == 'medium':
#                         description = f"moderate complexity (complexity={value:.3f})"
#                     else:
#                         description = f"simple pattern (complexity={value:.3f})"
#                 elif feature_name == 'energy':
#                     if level == 'very_high':
#                         description = f"very high energy (energy={value:.0f})"
#                     elif level == 'high':
#                         description = f"high energy (energy={value:.0f})"
#                     elif level == 'medium':
#                         description = f"moderate energy (energy={value:.0f})"
#                     else:
#                         description = f"low energy (energy={value:.0f})"
#                 elif feature_name == 'pitch_mean':
#                     if level == 'high':
#                         description = f"large head pitch angle (pitch={value:.3f})"
#                     elif level == 'medium':
#                         description = f"moderate head pitch angle (pitch={value:.3f})"
#                     else:
#                         description = f"small head pitch angle (pitch={value:.3f})"
#                 elif feature_name == 'roll_mean':
#                     if level == 'high':
#                         description = f"large head roll angle (roll={value:.3f})"
#                     elif level == 'medium':
#                         description = f"moderate head roll angle (roll={value:.3f})"
#                     else:
#                         description = f"small head roll angle (roll={value:.3f})"
#                 elif feature_name == 'peak_count':
#                     if level == 'high':
#                         description = f"many peaks (peak_count={value:.0f})"
#                     elif level == 'medium':
#                         description = f"moderate peaks (peak_count={value:.0f})"
#                     else:
#                         description = f"few peaks (peak_count={value:.0f})"
#                 elif feature_name == 'autocorr_peak_count':
#                     if level == 'high':
#                         description = f"many autocorrelation peaks (autocorr_peaks={value:.0f})"
#                     elif level == 'medium':
#                         description = f"moderate autocorrelation peaks (autocorr_peaks={value:.0f})"
#                     else:
#                         description = f"few autocorrelation peaks (autocorr_peaks={value:.0f})"
#                 elif feature_name == 'zero_crossings':
#                     if level == 'high':
#                         description = f"many zero crossings (zero_crossings={value:.0f})"
#                     elif level == 'medium':
#                         description = f"moderate zero crossings (zero_crossings={value:.0f})"
#                     else:
#                         description = f"few zero crossings (zero_crossings={value:.0f})"
#                 else:
#                     description = f"{feature_name}={value:.3f}"
                
#                 # 计算重要性分数（基于与中等水平的距离）
#                 if level == 'very_high':
#                     score = 3
#                 elif level == 'high':
#                     score = 2
#                 elif level == 'low':
#                     score = 1
#                 else:
#                     score = 0
                
#                 scored_features.append((feature_name, description, score))
        
#         # 按重要性分数排序，取前top_n
#         scored_features.sort(key=lambda x: x[2], reverse=True)
#         return scored_features[:top_n]

#     # 标签数字到行为名称的映射
#     LABEL_MAP = {
#         0: "Other",
#         1: "Rumination",
#         2: "Feeding"
#     }



#     def generate_enhanced_prompt(self, features: Dict[str, float], label: int, 
#                                 behavior_type: str, window_data: np.ndarray, 
#                                 window_labels: np.ndarray) -> str:
#         """
#         生成优化的Prompt
        
#         Args:
#             features: 特征字典
#             label: 行为标签
#             behavior_type: 行为类型
#             window_data: 窗口数据
#             window_labels: 窗口标签
            
#         Returns:
#             优化的Prompt字符串
#         """
#         # 直接生成数据摘要，使用所有13个核心特征
#         data_summary = self._generate_data_summary(window_data, features, window_labels)
#         # 统计窗口主标签和分布
#         if window_labels is not None and len(window_labels) > 0:
#             label_counter = Counter(window_labels)
#             main_label_num, main_label_count = label_counter.most_common(1)[0]
#             main_label_name = self.LABEL_MAP.get(main_label_num, str(main_label_num))
#             total = len(window_labels)
#             dist_str = ", ".join([
#                 f"{self.LABEL_MAP.get(lbl, str(lbl))}({cnt/total:.0%})"
#                 for lbl, cnt in label_counter.items()
#             ])
#             label_info = f"Window main label: {main_label_name} ({main_label_num}); Distribution: {dist_str}"
#         else:
#             label_info = "Window main label: Unknown; Distribution: Unknown"
#         # 传递给下一级
#         prompt = self._generate_optimized_prompt(label_info, data_summary, features)
#         return prompt.strip()



#     def _generate_data_summary(self, window_data: np.ndarray, features: Dict[str, float],
#                              window_labels: np.ndarray) -> str:
#         """生成行为区分性摘要"""
#         # 提取关键特征
#         periodicity = features.get('periodicity', 0)
#         pitch_mean = features.get('pitch_mean', 0)
#         pitch_std = features.get('pitch_std', 1e-6)
#         vm_mean = features.get('vm_mean', 0)
#         complexity = features.get('complexity', 0)
#         vm_std = features.get('vm_std', 0)
#         energy = features.get('energy', 0)
#         roll_mean = features.get('roll_mean', 0)
#         roll_std = features.get('roll_std', 0)
#         peak_count = features.get('peak_count', 0)
#         avg_peak_interval = features.get('avg_peak_interval', 0)
#         autocorr_peak_count = features.get('autocorr_peak_count', 0)
#         zero_crossings = features.get('zero_crossings', 0)

#         enhanced_indicators = []

#         # 1. 周期性特征描述
#         if periodicity > 0.8:
#             enhanced_indicators.append(f"very strong rhythmic pattern (periodicity={periodicity:.3f})")
#         elif periodicity > 0.6:
#             enhanced_indicators.append(f"strong rhythmic pattern (periodicity={periodicity:.3f})")
#         elif periodicity > 0.3:
#             enhanced_indicators.append(f"moderate rhythmic pattern (periodicity={periodicity:.3f})")
#         elif periodicity > 0.0:
#             enhanced_indicators.append(f"weak rhythmic pattern (periodicity={periodicity:.3f})")

#         # 2. 头部姿态描述
#         if abs(pitch_mean) > 0:
#             pitch_normalized = abs(pitch_mean) / (pitch_std + 1e-6)
#             if pitch_normalized > 2.0:
#                 if pitch_mean < 0:
#                     enhanced_indicators.append(f"significant head down (pitch={pitch_mean:.3f})")
#                 else:
#                     enhanced_indicators.append(f"significant head up (pitch={pitch_mean:.3f})")
#             elif pitch_normalized > 1.0:
#                 if pitch_mean < 0:
#                     enhanced_indicators.append(f"moderate head down (pitch={pitch_mean:.3f})")
#                 else:
#                     enhanced_indicators.append(f"moderate head up (pitch={pitch_mean:.3f})")

#         # 3. 运动强度描述（使用固定阈值）
#         if vm_mean > 2.5:
#             enhanced_indicators.append(f"very high intensity (vm={vm_mean:.3f})")
#         elif vm_mean > 1.5:
#             enhanced_indicators.append(f"high intensity (vm={vm_mean:.3f})")
#         elif vm_mean > 0.8:
#             enhanced_indicators.append(f"moderate intensity (vm={vm_mean:.3f})")
#         elif vm_mean > 0.0:
#             enhanced_indicators.append(f"low intensity (vm={vm_mean:.3f})")

#         # 4. 复杂度描述（使用固定阈值）
#         if complexity > 3.0:
#             enhanced_indicators.append(f"very complex pattern (complexity={complexity:.3f})")
#         elif complexity > 2.0:
#             enhanced_indicators.append(f"complex pattern (complexity={complexity:.3f})")
#         elif complexity > 1.0:
#             enhanced_indicators.append(f"moderate complexity (complexity={complexity:.3f})")
#         elif complexity > 0.0:
#             enhanced_indicators.append(f"simple pattern (complexity={complexity:.3f})")

#         # 5. 运动规律性描述（使用固定阈值）
#         if vm_std > 1.5:
#             enhanced_indicators.append(f"highly irregular movement (vm_std={vm_std:.3f})")
#         elif vm_std > 0.8:
#             enhanced_indicators.append(f"irregular movement (vm_std={vm_std:.3f})")
#         elif vm_std > 0.3:
#             enhanced_indicators.append(f"moderately regular movement (vm_std={vm_std:.3f})")
#         elif vm_std > 0.0:
#             enhanced_indicators.append(f"regular movement (vm_std={vm_std:.3f})")

#         # 6. 能量描述（使用固定阈值）
#         if energy > 1500:
#             enhanced_indicators.append(f"very high energy (energy={energy:.0f})")
#         elif energy > 800:
#             enhanced_indicators.append(f"high energy (energy={energy:.0f})")
#         elif energy > 300:
#             enhanced_indicators.append(f"moderate energy (energy={energy:.0f})")
#         elif energy > 0:
#             enhanced_indicators.append(f"low energy (energy={energy:.0f})")

#         # 7. 头部运动特征（使用固定阈值）
#         if abs(roll_mean) > 15:
#             enhanced_indicators.append(f"large head roll angle (roll={roll_mean:.3f})")
#         elif abs(roll_mean) > 5:
#             enhanced_indicators.append(f"moderate head roll angle (roll={roll_mean:.3f})")
#         elif abs(roll_mean) > 0:
#             enhanced_indicators.append(f"small head roll angle (roll={roll_mean:.3f})")

#         # 8. 峰值特征（使用固定阈值）
#         if peak_count > 7:
#             enhanced_indicators.append(f"many peaks (peak_count={peak_count:.0f})")
#         elif peak_count > 3:
#             enhanced_indicators.append(f"moderate peaks (peak_count={peak_count:.0f})")
#         elif peak_count > 0:
#             enhanced_indicators.append(f"few peaks (peak_count={peak_count:.0f})")

#         # 9. 自相关特征（使用固定阈值）
#         if autocorr_peak_count > 8:
#             enhanced_indicators.append(f"many autocorrelation peaks (autocorr_peaks={autocorr_peak_count:.0f})")
#         elif autocorr_peak_count > 5:
#             enhanced_indicators.append(f"moderate autocorrelation peaks (autocorr_peaks={autocorr_peak_count:.0f})")
#         elif autocorr_peak_count > 2:
#             enhanced_indicators.append(f"few autocorrelation peaks (autocorr_peaks={autocorr_peak_count:.0f})")

#         # 10. 过零点特征（使用固定阈值）
#         if zero_crossings > 15:
#             enhanced_indicators.append(f"many zero crossings (zero_crossings={zero_crossings:.0f})")
#         elif zero_crossings > 8:
#             enhanced_indicators.append(f"moderate zero crossings (zero_crossings={zero_crossings:.0f})")
#         elif zero_crossings > 3:
#             enhanced_indicators.append(f"few zero crossings (zero_crossings={zero_crossings:.0f})")

#         # 组合摘要（限制数量，选择最重要的特征）
#         if len(enhanced_indicators) > 6:
#             # 选择最重要的特征
#             important_features = self.select_important_features(features, top_n=6)
#             enhanced_indicators = [feature[1] for feature in important_features]

#         # 生成最终摘要
#         if enhanced_indicators:
#             summary = ", ".join(enhanced_indicators)
#         else:
#             if vm_mean > 0:
#                 summary = f"detectable movement activity (vm={vm_mean:.3f})"
#             else:
#                 summary = "minimal movement detected"
#         return summary


# #     def _generate_optimized_prompt(self, label_info: str, data_summary: str, features: Dict[str, float]) -> str:
# # #         examples = """Examples of concise behavioral descriptions:
# # # - FEEDING with head-down posture + moderate intensity → "Feeding behavior with head-down posture"
# # # - RUMINATION with strong periodicity + low intensity → "Rumination activity with rhythmic jaw movements"
# # # - OTHER with irregular patterns + high intensity → "Other activity with irregular movement patterns"
# # # """
# #         examples = """Examples of concise behavioral descriptions:
# # - FEEDING with low intensity + head position → "Moderate feeding behavior with head-down posture"
# # - RUMINATION with high periodicity + rhythmic → "Rhythmic movements characteristic of rumination behavior"
# # - OTHER with variable intensity + irregular → "Irregular patterns during other behavior phases"
# # """


# #         prompt = f'''
# # {examples}

# # {label_info}
# # Based on the examples above, analyze the following movement features and generate a concise behavioral description.
# # The description must clearly include the specific behavior type from the window label.

# # Requirements:
# # - Must include the exact behavior type name (feeding behavior/rumination behavior/other behavior) in the description but do not always place it at the beginning
# # - Follow the concise style of the examples above
# # - For FEEDING: focus on head posture and movement steadiness
# # - For RUMINATION: focus on rhythmic periodicity and repetitive motions
# # - For OTHER: focus on movement irregularity and intensity variations
# # - Use simple, direct behavioral language without commas
# # - Avoid complex adjectives and unnecessary details
# # - Generate 6-10 words that capture the core behavioral characteristics
# # - Do not use commas in the description
# # - Use simple phrases without punctuation

# # Movement feature summary: {data_summary}

# # Behavioral description:'''
# #         return prompt

#     def _generate_optimized_prompt(self, label_info: str, data_summary: str, features: Dict[str, float]) -> str:
#         examples = """Examples of concise behavioral descriptions:
# - FEEDING with head-down posture + moderate intensity → "Feeding behavior with head-down posture"
# - RUMINATION with strong periodicity + low intensity → "Rumination activity with rhythmic jaw movements"
# - OTHER with irregular patterns + high intensity → "Other activity with irregular movement patterns"
# """

#         prompt = f'''
# {examples}

# {label_info}
# Based on the examples above, analyze the following movement features and generate a concise behavioral description.
# The description must clearly include the specific behavior type from the window label.

# Requirements:
# - Must include the exact behavior type name (feeding/rumination/other) in the description
# - Follow the concise style of the examples above
# - For FEEDING: focus on head posture and movement steadiness
# - For RUMINATION: focus on rhythmic periodicity and repetitive motions
# - For OTHER: focus on movement irregularity and intensity variations
# - Use simple, direct behavioral language without commas
# - Avoid complex adjectives and unnecessary details
# - Generate 6-12 words that capture the core behavioral characteristics
# - Do not use commas in the description
# - Use simple phrases without punctuation

# Movement feature summary: {data_summary}

# Behavioral description:'''
#         return prompt


#     def call_gpt_api(self, prompt: str, window_data: np.ndarray, window_labels: np.ndarray, features: Dict[str, float] = None) -> str:
#         """
#         调用GPT API生成文本
#         """
#         max_bpe_tokens = 70
#         for attempt in range(self.max_retries):
#             try:
#                 # 动态调整max_tokens
#                 dynamic_max_tokens = min(self.max_tokens, 70)
#                 if hasattr(self, 'clip_model') and self.clip_model is not None:
#                     try:
#                         import clip
#                         base_tokens = clip.tokenize([prompt])[0]
#                         base_token_count = (base_tokens != 0).sum().item()
#                         dynamic_max_tokens = max(10, 77 - base_token_count - 5)
#                     except Exception as e:
#                         pass
#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": self.system_prompt},
#                         {"role": "user", "content": prompt}
#                     ],
#                     max_tokens=dynamic_max_tokens,
#                     temperature=self.temperature,
#                     timeout=self.timeout
#                 )
#                 generated_text = response.choices[0].message.content.strip()
#                 # 检查BPE分词后token数，超限则重试
#                 bpe_tokens = self.tokenizer.encode(generated_text)
#                 if len(bpe_tokens) <= max_bpe_tokens:
#                     return generated_text
#                 else:
#                     self.logger.warning(f"Generated text BPE tokens {len(bpe_tokens)} > {max_bpe_tokens}, retrying...")
#             except Exception as e:
#                 print(f"API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
#                 if attempt < self.max_retries - 1:
#                     time.sleep(2 ** attempt)  # 指数退避
#                 else:
#                     self.logger.warning(f"GPT API call failed, falling back to fallback generator")
#                     return self.fallback_generator.generate_text_for_window(window_data, window_labels)
#         # 多次重试后仍超限，降级到fallback生成器
#         self.logger.error(f"Failed to generate text within {max_bpe_tokens} BPE tokens after {self.max_retries} attempts. Falling back to fallback generator.")
#         return self.fallback_generator.generate_text_for_window(window_data, window_labels)
    
#     def generate_text_for_window(self, window_data: np.ndarray, window_labels: np.ndarray) -> str:
#         """
#         为单个窗口生成文本描述
        
#         Args:
#             window_data: 传感器数据窗口 [window_size, 3]
#             window_labels: 窗口内所有标签 [window_size]
            
#         Returns:
#             生成的文本描述
#         """
#         if self.use_gpt:
#             # GPT模式：使用GPT API生成文本
#             return self._generate_gpt_text(window_data, window_labels)
#         else:
#             # 普通模式：使用基于规则的文本生成器
#             return self.fallback_generator.generate_text_for_window(window_data, window_labels)
    
#     def _generate_gpt_text(self, window_data: np.ndarray, window_labels: np.ndarray) -> str:
#         """
#         使用GPT API生成文本描述
        
#         Args:
#             window_data: 传感器数据窗口
#             window_labels: 窗口标签
            
#         Returns:
#             GPT生成的文本描述
#         """
#         # 提取增强的窗口特征
#         features = self.extract_enhanced_features(window_data)

#         # 使用统一的Prompt生成
#         prompt = self.generate_enhanced_prompt(features, 0, "", window_data, window_labels)

#         # 调用GPT API，传递features避免重复提取
#         text_description = self.call_gpt_api(prompt, window_data, window_labels, features)
        
#         return text_description

#     def generate_text_for_window_with_features(self, window_data: np.ndarray, window_labels: np.ndarray, cached_features: Dict[str, float]) -> str:
#         """
#         使用缓存的特征为窗口生成文本描述（避免重复特征提取）

#         Args:
#             window_data: 传感器数据窗口
#             window_labels: 窗口标签
#             cached_features: 预先计算的特征

#         Returns:
#             生成的文本描述
#         """
#         if self.use_gpt:
#             # GPT模式：使用缓存的特征
#             prompt = self.generate_enhanced_prompt(cached_features, 0, "", window_data, window_labels)
#             return self.call_gpt_api(prompt, window_data, window_labels, cached_features)
#         else:
#             # 普通模式：使用基于规则的文本生成器
#             return self.fallback_generator.generate_text_for_window(window_data, window_labels)

#     def generate_text_for_window(self, window_data: np.ndarray, window_labels: np.ndarray) -> str:
#         """
#         为单个窗口生成文本描述

#         Args:
#             window_data: 传感器数据窗口 [window_size, 3]
#             window_labels: 窗口内所有标签 [window_size]

#         Returns:
#             生成的文本描述
#         """
#         if self.use_gpt:
#             # GPT模式：使用GPT API生成文本
#             return self._generate_gpt_text(window_data, window_labels)
#         else:
#             # 普通模式：使用基于规则的文本生成器
#             return self.fallback_generator.generate_text_for_window(window_data, window_labels)

#     def generate_texts_for_windows_batch(self, features: np.ndarray, labels: np.ndarray,
#                                        window_size: int, stride: int, batch_size: int = 32,
#                                        save_path: str = None) -> List[str]:
#         """
#         分批为所有窗口生成文本描述（内存优化版本）
        
#         Args:
#             features: 传感器数据
#             labels: 标签数据
#             window_size: 窗口大小
#             stride: 窗口步长
#             batch_size: 批处理大小
#             save_path: 保存路径（可选）
            
#         Returns:
#             文本描述列表
#         """
#         text_descriptions = []

#         if self.use_gpt:
#             print(f"Starting batch generation of window text descriptions using {self.model}...")
#             print("Mode: GPT API generation (batch processing)")
#         else:
#             print("Starting batch generation of window text descriptions using rule-based text generator...")
#             print("Mode: Rule-based generation (batch processing)")
#         print(f"Window size: {window_size}, Stride: {stride}, Batch size: {batch_size}")

#         # 计算窗口数量
#         num_windows = (len(features) - window_size) // stride + 1
#         print(f"Total windows: {num_windows}")

#         # 预先计算所有窗口的特征并缓存（避免重复计算）
#         print("提取所有窗口特征...")
#         all_features = {}  # 缓存所有窗口的特征
#         for window_idx in range(num_windows):
#             start_idx = window_idx * stride
#             end_idx = start_idx + window_size
#             window_data = features[start_idx:end_idx]
#             window_features = self.extract_enhanced_features(window_data)
#             all_features[window_idx] = window_features

#         # 使用固定阈值，无需计算统计分布
#         print("使用固定阈值进行特征分类...")
        
#         # 分批处理
#         for batch_start in range(0, num_windows, batch_size):
#             batch_end = min(batch_start + batch_size, num_windows)
#             batch_texts = []
            
#             print(f"Processing text batch {batch_start//batch_size + 1}/{(num_windows + batch_size - 1)//batch_size}")
            
#             for window_idx in range(batch_start, batch_end):
#                 start_idx = window_idx * stride
#                 end_idx = start_idx + window_size

#                 # 获取窗口数据
#                 window_data = features[start_idx:end_idx]
#                 window_labels = labels[start_idx:end_idx]

#                 # 使用缓存的特征生成文本描述
#                 cached_features = all_features[window_idx]
#                 text = self.generate_text_for_window_with_features(window_data, window_labels, cached_features)
#                 batch_texts.append(text)
                
#                 # 只在GPT模式下添加延迟（避免API限制）
#                 if self.use_gpt:
#                     time.sleep(0.1)
            
#             # 添加到总列表
#             text_descriptions.extend(batch_texts)
            
#             # 清理内存
#             del batch_texts
#             import gc
#             gc.collect()
        
#         print(f"Window text generation completed, generated {len(text_descriptions)} descriptions")
        
#         # 保存到CSV文件
#         if save_path:
#             # 将.txt后缀改为.csv
#             csv_path = save_path.replace('.txt', '.csv')
#             save_texts_to_csv(text_descriptions, csv_path)
        
#         return text_descriptions
    


















# # 行为名称不一定在文本开头
# import openai
# import time
# import logging
# from typing import List, Dict, Any, Tuple
# from collections import Counter
# import numpy as np
# from scipy import signal
# from fallback_text_generator import FallbackTextGenerator
# from simple_tokenizer import SimpleTokenizer
# from text_utils import save_texts_to_csv, load_texts_from_csv

# class GPTTextGenerator:
#     def __init__(self, args, clip_model=None):
#         """
#         初始化文本生成器
        
#         Args:
#             args: 配置参数（来自config.py的parse_args()）
#             clip_model: 可选，CLIP模型对象，用于动态token数估算
#         """
#         # 修正getattr参数
#         self.api_key = getattr(args, 'openai_api_key', None)
#         self.model = getattr(args, 'gpt_model', 'gpt-3.5-turbo')
#         self.use_gpt = getattr(args, 'use_gpt', False)
#         self.max_tokens = getattr(args, 'gpt_max_tokens', 70)
#         self.temperature = getattr(args, 'gpt_temperature', 0.5)
#         self.timeout = getattr(args, 'gpt_timeout', 30)
#         self.max_retries = getattr(args, 'gpt_max_retries', 3)
#         self.system_prompt = getattr(args, 'gpt_system_prompt', "")
#         self.logger = logging.getLogger(__name__)
#         self.base_url = getattr(args, 'gpt_base_url', None)
        
#         # 初始化tokenizer用于精确token计数（使用BPE分词）
#         self.tokenizer = SimpleTokenizer(bpe_path="bpe_simple_vocab_16e6.txt.gz")
        
#         # 初始化普通文本生成器
#         self.fallback_generator = FallbackTextGenerator(max_tokens=self.max_tokens)
        
#         # 根据模式进行初始化
#         if self.use_gpt:
#             # GPT模式
#             if self.api_key:
#                 if self.base_url:
#                     self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
#                 else:
#                     self.client = openai.OpenAI(api_key=self.api_key)
#                 self.logger.info(f"Using GPT mode, model: {self.model}, max tokens: {self.max_tokens}, base_url: {self.base_url}")
#             else:
#                 self.logger.warning("GPT mode requires API key, switching to normal mode")
#                 self.use_gpt = False
#         else:
#             # 普通模式
#             self.logger.info(f"Using rule-based text generation mode, max tokens: {self.max_tokens}")
        
#         self.clip_model = clip_model

#         # 特征统计信息存储
#         self.feature_stats = {}
#         self.stats_computed = False

#     def extract_enhanced_features(self, window_data: np.ndarray) -> Dict[str, float]:
#         """
#         直接提取12个区分性特征（专门针对三种行为区分）
        
#         Args:
#             window_data: 传感器数据窗口
            
#         Returns:
#             12个区分性特征字典
#         """
#         if window_data.ndim == 1:
#             window_data = window_data.reshape(-1, 1)
#         # 分离三轴数据
#         x_data = window_data[:, 0]  # 前后方向
#         y_data = window_data[:, 1]  # 左右方向
#         z_data = window_data[:, 2]  # 上下方向
        
#         # 计算合成加速度
#         vm = np.sqrt(x_data**2 + y_data**2 + z_data**2)
        
#         # 1. 周期性特征（反刍的关键特征）
#         # 自相关分析
#         autocorr = np.correlate(vm, vm, mode='full')
#         autocorr = autocorr[len(autocorr)//2:]
#         autocorr_peaks, _ = signal.find_peaks(autocorr[:len(autocorr)//2])
#         autocorr_peak_count = len(autocorr_peaks)
        
#         # 周期性强度
#         if len(autocorr) > 1:
#             periodicity = np.max(autocorr[1:]) / autocorr[0]
#         else:
#             periodicity = 0
        
#         # 2. 头部姿态特征（进食的关键特征）
#         # 俯仰角（前后倾斜）
#         pitch_angles = np.arctan2(x_data, np.sqrt(y_data**2 + z_data**2))
#         pitch_mean = np.mean(pitch_angles)
#         pitch_std = np.std(pitch_angles)
            
#         # 横滚角（左右倾斜）
#         roll_angles = np.arctan2(y_data, np.sqrt(x_data**2 + z_data**2))
#         roll_mean = np.mean(roll_angles)
#         roll_std = np.std(roll_angles)
            
#         # 3. 运动强度特征（其他活动的关键特征）
#         vm_mean = np.mean(vm)
#         vm_std = np.std(vm)
#         energy = np.sum(vm**2)
        
#         # 4. 时序规律性特征
#         peaks, _ = signal.find_peaks(vm)
#         peak_count = len(peaks)
        
#         if len(peaks) > 1:
#             peak_intervals = np.diff(peaks)
#             avg_peak_interval = np.mean(peak_intervals)
#         else:
#             avg_peak_interval = len(vm)
        
#         # 5. 运动复杂度特征
#         # # 基于熵的复杂度
#         # hist, _ = np.histogram(vm, bins=20)
#         # complexity = -np.sum(hist * np.log(hist + 1e-10))
#         # 基于熵的复杂度
#         hist, _ = np.histogram(vm, bins=20)
#         # 归一化为概率分布
#         prob = hist / np.sum(hist)
#         # 只对非零概率计算熵，避免log(0)
#         prob = prob[prob > 0]
#         if len(prob) > 0:
#             complexity = -np.sum(prob * np.log(prob))
#         else:
#             complexity = 0
        
#         # 过零点数
#         zero_crossings = np.sum(np.diff(np.signbit(vm - np.mean(vm))) != 0)
        
#         return {
#             # 周期性特征（反刍关键）
#             'periodicity': periodicity,
#             'autocorr_peak_count': autocorr_peak_count,
            
#             # 头部姿态特征（进食关键）
#             'pitch_mean': pitch_mean,
#             'pitch_std': pitch_std,
#             'roll_mean': roll_mean,
#             'roll_std': roll_std,
            
#             # 运动强度特征（其他活动关键）
#             'vm_mean': vm_mean,
#             'vm_std': vm_std,
#             'energy': energy,
            
#             # 时序规律性特征
#             'peak_count': peak_count,
#             'avg_peak_interval': avg_peak_interval,
            
#             # 运动复杂度特征
#             'complexity': complexity,
#             'zero_crossings': zero_crossings
#         }

#     def compute_feature_statistics(self, all_features_list):
#         """
#         计算特征的统计分布，用于动态阈值

#         Args:
#             all_features_list: 所有窗口的特征字典列表
#         """
#         if not all_features_list:
#             return

#         # 计算关键特征的统计分布
#         key_features = ['periodicity', 'vm_mean', 'complexity', 'pitch_mean', 'energy']

#         for feature_name in key_features:
#             values = [f[feature_name] for f in all_features_list if feature_name in f]
#             if values:
#                 self.feature_stats[feature_name] = {
#                     'mean': np.mean(values),
#                     'std': np.std(values),
#                     'p25': np.percentile(values, 25),
#                     'p50': np.percentile(values, 50),
#                     'p75': np.percentile(values, 75),
#                     'p80': np.percentile(values, 80),
#                     'p85': np.percentile(values, 85),
#                     'p90': np.percentile(values, 90)
#                 }

#         self.stats_computed = True
#         print(f"特征统计计算完成，涵盖 {len(key_features)} 个关键特征")


#     # 标签数字到行为名称的映射
#     LABEL_MAP = {
#         0: "Other",
#         1: "Rumination",
#         2: "Feeding"
#     }



#     def generate_enhanced_prompt(self, features: Dict[str, float], label: int, 
#                                 behavior_type: str, window_data: np.ndarray, 
#                                 window_labels: np.ndarray) -> str:
#         """
#         生成优化的Prompt
        
#         Args:
#             features: 特征字典
#             label: 行为标签（不再使用）
#             behavior_type: 行为类型（不再使用）
#             window_data: 窗口数据
#             window_labels: 窗口标签
            
#         Returns:
#             优化的Prompt字符串
#         """
#         # 直接生成数据摘要，使用所有13个核心特征
#         data_summary = self._generate_data_summary(window_data, features, window_labels)
#         # 统计窗口主标签和分布
#         if window_labels is not None and len(window_labels) > 0:
#             label_counter = Counter(window_labels)
#             main_label_num, main_label_count = label_counter.most_common(1)[0]
#             main_label_name = self.LABEL_MAP.get(main_label_num, str(main_label_num))
#             total = len(window_labels)
#             dist_str = ", ".join([
#                 f"{self.LABEL_MAP.get(lbl, str(lbl))}({cnt/total:.0%})"
#                 for lbl, cnt in label_counter.items()
#             ])
#             label_info = f"Window main label: {main_label_name} ({main_label_num}); Distribution: {dist_str}"
#         else:
#             label_info = "Window main label: Unknown; Distribution: Unknown"
#         # 传递给下一级
#         prompt = self._generate_optimized_prompt(label_info, data_summary, features)
#         return prompt.strip()



#     def _generate_data_summary(self, window_data: np.ndarray, features: Dict[str, float],
#                              window_labels: np.ndarray) -> str:
#         """生成行为区分性摘要"""
#         # 提取关键特征
#         periodicity = features.get('periodicity', 0)
#         pitch_mean = features.get('pitch_mean', 0)
#         pitch_std = features.get('pitch_std', 0)
#         vm_mean = features.get('vm_mean', 0)
#         complexity = features.get('complexity', 0)

#         enhanced_indicators = []

#         # 1. 周期性特征描述（使用动态阈值，放宽标准）
#         if self.stats_computed and 'periodicity' in self.feature_stats:
#             stats = self.feature_stats['periodicity']
#             if periodicity > stats['p90']:
#                 enhanced_indicators.append(f"very strong rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > stats['p80']:
#                 enhanced_indicators.append(f"strong rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > stats['p50']:
#                 enhanced_indicators.append(f"moderate rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > stats['p25']:
#                 enhanced_indicators.append(f"weak rhythmic pattern (periodicity={periodicity:.3f})")
#         else:
#             # 回退到固定阈值
#             if periodicity > 0.6:
#                 enhanced_indicators.append(f"very strong rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > 0.4:
#                 enhanced_indicators.append(f"strong rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > 0.2:
#                 enhanced_indicators.append(f"moderate rhythmic pattern (periodicity={periodicity:.3f})")
#             elif periodicity > 0:
#                 enhanced_indicators.append(f"weak rhythmic pattern (periodicity={periodicity:.3f})")

#         # 2. 头部姿态描述
#         if abs(pitch_mean) > 0:
#             pitch_normalized = abs(pitch_mean) / (pitch_std + 1e-6)
#             if pitch_normalized > 2.0:
#                 if pitch_mean < 0:
#                     enhanced_indicators.append(f"significant head down (pitch={pitch_mean:.3f})")
#                 else:
#                     enhanced_indicators.append(f"significant head up (pitch={pitch_mean:.3f})")
#             elif pitch_normalized > 1.0:
#                 if pitch_mean < 0:
#                     enhanced_indicators.append(f"moderate head down (pitch={pitch_mean:.3f})")
#                 else:
#                     enhanced_indicators.append(f"moderate head up (pitch={pitch_mean:.3f})")

#         # 3. 运动强度描述（使用动态阈值，放宽标准）
#         if self.stats_computed and 'vm_mean' in self.feature_stats:
#             stats = self.feature_stats['vm_mean']
#             if vm_mean > stats['p85']:
#                 enhanced_indicators.append(f"high intensity (vm={vm_mean:.3f})")
#             elif vm_mean > stats['p50']:
#                 enhanced_indicators.append(f"moderate intensity (vm={vm_mean:.3f})")
#             elif vm_mean > 0:
#                 enhanced_indicators.append(f"low intensity (vm={vm_mean:.3f})")
#         else:
#             # 回退到固定阈值
#             if vm_mean > 1.5:
#                 enhanced_indicators.append(f"high intensity (vm={vm_mean:.3f})")
#             elif vm_mean > 0.8:
#                 enhanced_indicators.append(f"moderate intensity (vm={vm_mean:.3f})")
#             elif vm_mean > 0:
#                 enhanced_indicators.append(f"low intensity (vm={vm_mean:.3f})")

#         # 4. 复杂度描述（使用动态阈值，放宽标准）
#         if self.stats_computed and 'complexity' in self.feature_stats:
#             stats = self.feature_stats['complexity']
#             if complexity > stats['p85']:
#                 enhanced_indicators.append(f"very complex pattern (complexity={complexity:.3f})")
#             elif complexity > stats['p50']:
#                 enhanced_indicators.append(f"complex pattern (complexity={complexity:.3f})")
#             elif complexity > 0:
#                 enhanced_indicators.append(f"simple pattern (complexity={complexity:.3f})")
#         else:
#             # 回退到固定阈值
#             if complexity > 3.0:
#                 enhanced_indicators.append(f"very complex pattern (complexity={complexity:.3f})")
#             elif complexity > 2.0:
#                 enhanced_indicators.append(f"complex pattern (complexity={complexity:.3f})")
#             elif complexity > 0:
#                 enhanced_indicators.append(f"simple pattern (complexity={complexity:.3f})")

#         # 组合摘要（限制数量）
#         if len(enhanced_indicators) > 4:
#             enhanced_indicators = enhanced_indicators[:4]

#         # 生成最终摘要
#         if enhanced_indicators:
#             summary = ", ".join(enhanced_indicators)
#         else:
#             if vm_mean > 0:
#                 summary = f"detectable movement activity (vm={vm_mean:.3f})"
#             else:
#                 summary = "minimal movement detected"
#         return summary


#     def _generate_optimized_prompt(self, label_info: str, data_summary: str, features: Dict[str, float]) -> str:
# #         examples = """Examples of concise behavioral descriptions:
# # - FEEDING with head-down posture + moderate intensity → "Feeding behavior with head-down posture"
# # - RUMINATION with strong periodicity + low intensity → "Rumination activity with rhythmic jaw movements"
# # - OTHER with irregular patterns + high intensity → "Other activity with irregular movement patterns"
# # """
#         examples = """Examples of concise behavioral descriptions:
# - FEEDING with low intensity + head position → "Moderate feeding behavior with head-down posture"
# - RUMINATION with high periodicity + rhythmic → "Rhythmic movements characteristic of rumination behavior"
# - OTHER with variable intensity + irregular → "Irregular patterns during other behavior phases"
# """


#         prompt = f'''
# {examples}

# {label_info}
# Based on the examples above, analyze the following movement features and generate a concise behavioral description.
# The description must clearly include the specific behavior type from the window label.

# Requirements:
# - Must include the exact behavior type name (feeding behavior/rumination behavior/other behavior) in the description but do not always place it at the beginning
# - Follow the concise style of the examples above
# - For FEEDING: focus on head posture and movement steadiness
# - For RUMINATION: focus on rhythmic periodicity and repetitive motions
# - For OTHER: focus on movement irregularity and intensity variations
# - Use simple, direct behavioral language without commas
# - Avoid complex adjectives and unnecessary details
# - Generate 6-12 words that capture the core behavioral characteristics
# - Use simple phrases without punctuation

# Movement feature summary: {data_summary}

# Behavioral description:'''
#         return prompt




#     def call_gpt_api(self, prompt: str, window_data: np.ndarray, window_labels: np.ndarray, features: Dict[str, float] = None) -> str:
#         """
#         调用GPT API生成文本
#         """
#         max_bpe_tokens = 70
#         for attempt in range(self.max_retries):
#             try:
#                 # 动态调整max_tokens
#                 dynamic_max_tokens = min(self.max_tokens, 70)
#                 if hasattr(self, 'clip_model') and self.clip_model is not None:
#                     try:
#                         import clip
#                         base_tokens = clip.tokenize([prompt])[0]
#                         base_token_count = (base_tokens != 0).sum().item()
#                         dynamic_max_tokens = max(10, 77 - base_token_count - 5)
#                     except Exception as e:
#                         pass
#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": self.system_prompt},
#                         {"role": "user", "content": prompt}
#                     ],
#                     max_tokens=dynamic_max_tokens,
#                     temperature=self.temperature,
#                     timeout=self.timeout
#                 )
#                 generated_text = response.choices[0].message.content.strip()
#                 # 检查BPE分词后token数，超限则重试
#                 bpe_tokens = self.tokenizer.encode(generated_text)
#                 if len(bpe_tokens) <= max_bpe_tokens:
#                     return generated_text
#                 else:
#                     self.logger.warning(f"Generated text BPE tokens {len(bpe_tokens)} > {max_bpe_tokens}, retrying...")
#             except Exception as e:
#                 print(f"API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
#                 if attempt < self.max_retries - 1:
#                     time.sleep(2 ** attempt)  # 指数退避
#                 else:
#                     self.logger.warning(f"GPT API call failed, falling back to fallback generator")
#                     return self.fallback_generator.generate_text_for_window(window_data, window_labels)
#         # 多次重试后仍超限，降级到fallback生成器
#         self.logger.error(f"Failed to generate text within {max_bpe_tokens} BPE tokens after {self.max_retries} attempts. Falling back to fallback generator.")
#         return self.fallback_generator.generate_text_for_window(window_data, window_labels)
    
#     def generate_text_for_window(self, window_data: np.ndarray, window_labels: np.ndarray) -> str:
#         """
#         为单个窗口生成文本描述
        
#         Args:
#             window_data: 传感器数据窗口 [window_size, 3]
#             window_labels: 窗口内所有标签 [window_size]
            
#         Returns:
#             生成的文本描述
#         """
#         if self.use_gpt:
#             # GPT模式：使用GPT API生成文本
#             return self._generate_gpt_text(window_data, window_labels)
#         else:
#             # 普通模式：使用基于规则的文本生成器
#             return self.fallback_generator.generate_text_for_window(window_data, window_labels)
    
#     def _generate_gpt_text(self, window_data: np.ndarray, window_labels: np.ndarray) -> str:
#         """
#         使用GPT API生成文本描述
        
#         Args:
#             window_data: 传感器数据窗口
#             window_labels: 窗口标签
            
#         Returns:
#             GPT生成的文本描述
#         """
#         # 提取增强的窗口特征
#         features = self.extract_enhanced_features(window_data)

#         # 使用统一的Prompt生成
#         prompt = self.generate_enhanced_prompt(features, 0, "", window_data, window_labels)

#         # 调用GPT API，传递features避免重复提取
#         text_description = self.call_gpt_api(prompt, window_data, window_labels, features)
        
#         return text_description

#     def generate_text_for_window_with_features(self, window_data: np.ndarray, window_labels: np.ndarray, cached_features: Dict[str, float]) -> str:
#         """
#         使用缓存的特征为窗口生成文本描述（避免重复特征提取）

#         Args:
#             window_data: 传感器数据窗口
#             window_labels: 窗口标签
#             cached_features: 预先计算的特征

#         Returns:
#             生成的文本描述
#         """
#         if self.use_gpt:
#             # GPT模式：使用缓存的特征
#             prompt = self.generate_enhanced_prompt(cached_features, 0, "", window_data, window_labels)
#             return self.call_gpt_api(prompt, window_data, window_labels, cached_features)
#         else:
#             # 普通模式：使用基于规则的文本生成器
#             return self.fallback_generator.generate_text_for_window(window_data, window_labels)

#     def generate_text_for_window(self, window_data: np.ndarray, window_labels: np.ndarray) -> str:
#         """
#         为单个窗口生成文本描述

#         Args:
#             window_data: 传感器数据窗口 [window_size, 3]
#             window_labels: 窗口内所有标签 [window_size]

#         Returns:
#             生成的文本描述
#         """
#         if self.use_gpt:
#             # GPT模式：使用GPT API生成文本
#             return self._generate_gpt_text(window_data, window_labels)
#         else:
#             # 普通模式：使用基于规则的文本生成器
#             return self.fallback_generator.generate_text_for_window(window_data, window_labels)

#     def generate_texts_for_windows_batch(self, features: np.ndarray, labels: np.ndarray,
#                                        window_size: int, stride: int, batch_size: int = 32,
#                                        save_path: str = None) -> List[str]:
#         """
#         分批为所有窗口生成文本描述（内存优化版本）
        
#         Args:
#             features: 传感器数据
#             labels: 标签数据
#             window_size: 窗口大小
#             stride: 窗口步长
#             batch_size: 批处理大小
#             save_path: 保存路径（可选）
            
#         Returns:
#             文本描述列表
#         """
#         text_descriptions = []

#         if self.use_gpt:
#             print(f"Starting batch generation of window text descriptions using {self.model}...")
#             print("Mode: GPT API generation (batch processing)")
#         else:
#             print("Starting batch generation of window text descriptions using rule-based text generator...")
#             print("Mode: Rule-based generation (batch processing)")
#         print(f"Window size: {window_size}, Stride: {stride}, Batch size: {batch_size}")

#         # 计算窗口数量
#         num_windows = (len(features) - window_size) // stride + 1
#         print(f"Total windows: {num_windows}")

#         # 预先计算所有窗口的特征并缓存（避免重复计算）
#         print("提取所有窗口特征...")
#         all_features = {}  # 缓存所有窗口的特征
#         for window_idx in range(num_windows):
#             start_idx = window_idx * stride
#             end_idx = start_idx + window_size
#             window_data = features[start_idx:end_idx]
#             window_features = self.extract_enhanced_features(window_data)
#             all_features[window_idx] = window_features

#         # 计算特征统计分布（用于动态阈值）
#         if not self.stats_computed:
#             print("计算特征统计分布...")
#             self.compute_feature_statistics(list(all_features.values()))
#             print("特征统计计算完成")
        
#         # 分批处理
#         for batch_start in range(0, num_windows, batch_size):
#             batch_end = min(batch_start + batch_size, num_windows)
#             batch_texts = []
            
#             print(f"Processing text batch {batch_start//batch_size + 1}/{(num_windows + batch_size - 1)//batch_size}")
            
#             for window_idx in range(batch_start, batch_end):
#                 start_idx = window_idx * stride
#                 end_idx = start_idx + window_size

#                 # 获取窗口数据
#                 window_data = features[start_idx:end_idx]
#                 window_labels = labels[start_idx:end_idx]

#                 # 使用缓存的特征生成文本描述
#                 cached_features = all_features[window_idx]
#                 text = self.generate_text_for_window_with_features(window_data, window_labels, cached_features)
#                 batch_texts.append(text)
                
#                 # 只在GPT模式下添加延迟（避免API限制）
#                 if self.use_gpt:
#                     time.sleep(0.1)
            
#             # 添加到总列表
#             text_descriptions.extend(batch_texts)
            
#             # 清理内存
#             del batch_texts
#             import gc
#             gc.collect()
        
#         print(f"Window text generation completed, generated {len(text_descriptions)} descriptions")
        
#         # 保存到CSV文件
#         if save_path:
#             # 将.txt后缀改为.csv
#             csv_path = save_path.replace('.txt', '.csv')
#             save_texts_to_csv(text_descriptions, csv_path)
        
#         return text_descriptions
    