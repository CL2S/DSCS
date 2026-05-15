import json
import math
from html import escape
from pathlib import Path
from typing import Dict, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "forecasting"
STATS_PATH = OUTPUT_DIR / "formation_feature_stats.json"
HTML_PATH = OUTPUT_DIR / "formation_feature_report.html"

DATASET_CONFIGS = [
    {
        "name": "tourism_monthly",
        "path": ROOT / "input" / "forecasting" / "tourism_monthly" / "tourism_monthly_dataset.tsf",
        "history_length": 15,
        "forecast_horizon": 24,
        "seasonality": 12,
        "profile": "366 条月度旅游序列。窗口只有 15 个月、但要预测未来 24 个月，所以 formation 更像在问“当前窗口是什么经营阶段、季节模板有多可靠”。",
        "color": "#0f766e",
    },
    {
        "name": "nn5_daily",
        "path": ROOT / "input" / "forecasting" / "nn5_daily" / "nn5_daily_dataset_without_missing_values.tsf",
        "history_length": 9,
        "forecast_horizon": 56,
        "seasonality": 7,
        "profile": "111 条日度序列。历史窗口只有 9 天，却要外推 56 天，特征会更强调短期斜率、突变、以及“最近这几天到底是上升还是回落”。",
        "color": "#ea580c",
    },
    {
        "name": "australian_electricity_demand",
        "path": ROOT / "input" / "forecasting" / "australian_electricity_demand" / "australian_electricity_demand_dataset.tsf",
        "history_length": 420,
        "forecast_horizon": 336,
        "seasonality": 48,
        "profile": "5 条半小时级负荷序列。窗口长达 420 步，日周期很强，所以 formation 更像在刻画“当前处在日内哪个相位、曲线是否偏离稳定负荷形状”。",
        "color": "#2563eb",
    },
]

FEATURE_ORDER = [
    "local_slope",
    "medium_slope",
    "local_volatility",
    "volatility_ratio",
    "seasonal_gap_norm",
    "seasonal_strength",
    "level_shift_norm",
    "max_zscore",
    "range_norm",
    "phase_sin",
    "phase_cos",
    "change_proxy",
    "curvature",
    "autocorr_lag1",
    "stability_score",
    "regime_mix_score",
]

FEATURE_SPECS = {
    "local_slope": {
        "title": "1. local_slope",
        "formula": "(x_T - x_1) / ((T - 1) * scale)",
        "intuition": "看整个窗口首尾连线的斜率。正值表示窗口末端比开头更高，负值表示在回落，绝对值越大说明短窗方向越明确。",
    },
    "medium_slope": {
        "title": "2. medium_slope",
        "formula": "(x_T - x_{T-k+1}) / ((k - 1) * scale),  k = max(2, floor(T / 2))",
        "intuition": "只盯窗口后半段的斜率，比 local_slope 更关注“最近这半段正在怎么走”。如果它和 local_slope 方向相反，通常意味着窗口里发生了转向。",
    },
    "local_volatility": {
        "title": "3. local_volatility",
        "formula": "sqrt(mean((x_t - center)^2)) / scale",
        "intuition": "代码里的 scale 本身就是这个波动尺度的近似，所以它基本固定在 1。这个量更像归一化基准，而不是用来区分数据集的主特征。",
    },
    "volatility_ratio": {
        "title": "4. volatility_ratio",
        "formula": "tail_volatility / full_window_volatility",
        "intuition": "比较窗口尾部三分之一和整窗的波动强度。大于 1 说明最近更躁动，小于 1 说明尾部反而平静下来了。",
    },
    "seasonal_gap_norm": {
        "title": "5. seasonal_gap_norm",
        "formula": "(x_T - x_{T-s}) / scale",
        "intuition": "拿当前点和上一个季节位置做对比。正值表示“同一季相下今年/本轮更高”，负值表示更低。",
    },
    "seasonal_strength": {
        "title": "6. seasonal_strength",
        "formula": "1 - min(2, mean(|recent_block - previous_block|) / scale) / 2",
        "intuition": "比较最近一个 season block 和前一个 season block 是否像。越接近 1，说明季节模板越稳定复现；越接近 0，说明周期形状已经变了。",
    },
    "level_shift_norm": {
        "title": "7. level_shift_norm",
        "formula": "(mean(last_quarter) - mean(first_quarter)) / scale",
        "intuition": "比较窗口前四分之一和后四分之一的平均水平。它不是瞬时趋势，而是“这一整段是否抬升/下沉了一个台阶”。",
    },
    "max_zscore": {
        "title": "8. max_zscore",
        "formula": "max_t |(x_t - center) / scale|",
        "intuition": "衡量窗口里最突出的那个点离均值有多远。越大越容易出现尖峰、深坑或极端偏移。",
    },
    "range_norm": {
        "title": "9. range_norm",
        "formula": "(max(x) - min(x)) / scale",
        "intuition": "窗口的峰谷跨度。它不看顺序，只看振幅有多大。",
    },
    "phase_sin": {
        "title": "10. phase_sin",
        "formula": "sin(2π * (end_index mod s) / s)",
        "intuition": "把窗口终点映射到季节周期里的圆周位置。它不表示强弱，只表示“当前在周期的哪个相位”。",
    },
    "phase_cos": {
        "title": "11. phase_cos",
        "formula": "cos(2π * (end_index mod s) / s)",
        "intuition": "和 phase_sin 成对出现，用二维相位编码避免月份/星期的头尾断裂问题。",
    },
    "change_proxy": {
        "title": "12. change_proxy",
        "formula": "min(max_t |x_t - x_{t-1}| / scale, 5) / 5",
        "intuition": "抓窗口里最剧烈的单步跳变。越大说明只要一步就能跳出明显台阶或尖峰。",
    },
    "curvature": {
        "title": "13. curvature",
        "formula": "mean(x_t - 2x_{t-1} + x_{t-2}) / scale",
        "intuition": "看二阶差分，也就是弯曲程度。正值更像向上拱或加速上行，负值更像向下拱或加速下行。",
    },
    "autocorr_lag1": {
        "title": "14. autocorr_lag1",
        "formula": "corr(x_{1:T-1}, x_{2:T})",
        "intuition": "相邻时刻像不像。接近 1 说明序列很平滑、一步一步连续延伸；接近 0 或负值说明相邻点更跳、更噪。",
    },
    "stability_score": {
        "title": "15. stability_score",
        "formula": "1 / (1 + volatility + 0.5 * change_proxy + |level_shift_norm|)",
        "intuition": "把波动、突变、台阶偏移合成一个稳定性分数。越大越稳，越小越不稳。",
    },
    "regime_mix_score": {
        "title": "16. regime_mix_score",
        "formula": "min(4, |local_slope| + 0.75|medium_slope| + volatility + |level_shift| + change_proxy) / 4",
        "intuition": "把趋势、波动、台阶和突变揉成一个“状态混合度”。越高表示一个窗口里同时混着更多行为成分。",
    },
}


def convert_tsf_to_rows(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    found_data = False
    with path.open("r", encoding="cp1252") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                if line.startswith("@data"):
                    found_data = True
                continue
            if not found_data:
                continue
            parts = line.split(":")
            series_values: List[float] = []
            for value in parts[-1].split(","):
                if value == "?":
                    continue
                numeric = float(value)
                if not math.isnan(numeric):
                    series_values.append(numeric)
            rows.append(series_values)
    return rows


def safe_scale(values: Sequence[float]) -> float:
    mean_value = sum(values) / max(1, len(values))
    variance = sum((value - mean_value) ** 2 for value in values) / max(1, len(values))
    std_value = math.sqrt(max(variance, 1e-8))
    fallback = max(1.0, abs(mean_value), max(abs(value) for value in values))
    return max(std_value, fallback * 1e-2)


def slope(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return (values[-1] - values[0]) / max(1, len(values) - 1)


def phase_features(end_index: int, seasonality: int) -> tuple[float, float]:
    if seasonality <= 1:
        return 0.0, 1.0
    phase = 2.0 * math.pi * (end_index % seasonality) / seasonality
    return math.sin(phase), math.cos(phase)


def max_zscore(values: Sequence[float], center: float, scale: float) -> float:
    if not values:
        return 0.0
    return max(abs((value - center) / max(scale, 1e-6)) for value in values)


def autocorr_lag1(values: Sequence[float]) -> float:
    if len(values) <= 2:
        return 0.0
    left = values[:-1]
    right = values[1:]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((left_value - left_mean) * (right_value - right_mean) for left_value, right_value in zip(left, right))
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    return numerator / math.sqrt(max(left_var * right_var, 1e-8))


def curvature(values: Sequence[float], scale: float) -> float:
    if len(values) <= 2:
        return 0.0
    second_diffs = [
        values[index] - 2.0 * values[index - 1] + values[index - 2]
        for index in range(2, len(values))
    ]
    return (sum(second_diffs) / max(1, len(second_diffs))) / max(scale, 1e-6)


def build_window_formation(context: Sequence[float], seasonality: int, end_index: int) -> List[float]:
    scale = safe_scale(context)
    center = sum(context) / max(1, len(context))
    local_slope = slope(context) / max(scale, 1e-6)
    mid_point = max(2, len(context) // 2)
    medium_slope = slope(context[-mid_point:]) / max(scale, 1e-6)
    volatility = math.sqrt(sum((value - center) ** 2 for value in context) / max(1, len(context))) / max(scale, 1e-6)
    tail_width = max(2, len(context) // 3)
    tail_values = context[-tail_width:]
    tail_center = sum(tail_values) / max(1, tail_width)
    tail_volatility = math.sqrt(sum((value - tail_center) ** 2 for value in tail_values) / max(1, tail_width)) / max(scale, 1e-6)
    volatility_ratio = tail_volatility / max(volatility, 1e-6)

    seasonal_gap = 0.0
    seasonal_strength = 0.0
    if seasonality > 0 and len(context) > seasonality:
        seasonal_gap = (context[-1] - context[-seasonality]) / max(scale, 1e-6)
        recent_block = context[-seasonality:]
        previous_block = context[-2 * seasonality : -seasonality]
        if previous_block:
            seasonal_strength = 1.0 - min(
                2.0,
                sum(abs(left - right) for left, right in zip(recent_block, previous_block))
                / max(1, len(previous_block))
                / max(scale, 1e-6),
            ) / 2.0
            seasonal_strength = max(0.0, seasonal_strength)

    chunk = max(1, len(context) // 4)
    left_mean = sum(context[:chunk]) / max(1, chunk)
    right_mean = sum(context[-chunk:]) / max(1, chunk)
    level_shift_norm = (right_mean - left_mean) / max(scale, 1e-6)
    range_norm = (max(context) - min(context)) / max(scale, 1e-6)
    maximum_z = max_zscore(context, center, scale)
    phase_sin, phase_cos = phase_features(end_index, seasonality)

    diffs = [abs(context[index] - context[index - 1]) for index in range(1, len(context))]
    change_proxy = (max(diffs) / max(scale, 1e-6)) if diffs else 0.0
    change_proxy = min(change_proxy, 5.0) / 5.0

    local_curvature = curvature(context, scale)
    lag1_corr = autocorr_lag1(context)
    stability_score = 1.0 / (1.0 + volatility + 0.5 * change_proxy + abs(level_shift_norm))
    regime_mix_score = min(
        4.0,
        abs(local_slope) + 0.75 * abs(medium_slope) + volatility + abs(level_shift_norm) + change_proxy,
    ) / 4.0

    return [
        local_slope,
        medium_slope,
        volatility,
        volatility_ratio,
        seasonal_gap,
        seasonal_strength,
        level_shift_norm,
        maximum_z,
        range_norm,
        phase_sin,
        phase_cos,
        change_proxy,
        local_curvature,
        lag1_corr,
        stability_score,
        regime_mix_score,
    ]


def quantiles(values: Sequence[float], quantile_list: Sequence[float]) -> Dict[str, float]:
    sorted_values = sorted(values)
    size = len(sorted_values)
    summary: Dict[str, float] = {}
    for quantile in quantile_list:
        if size == 1:
            summary[f"q{int(quantile * 100):02d}"] = float(sorted_values[0])
            continue
        position = quantile * (size - 1)
        low = int(position)
        high = min(low + 1, size - 1)
        ratio = position - low
        interpolated = sorted_values[low] * (1.0 - ratio) + sorted_values[high] * ratio
        summary[f"q{int(quantile * 100):02d}"] = float(interpolated)
    return summary


def collect_dataset_feature_stats(config: Dict[str, object], max_train_windows_per_series: int = 10) -> Dict[str, object]:
    series_collection = convert_tsf_to_rows(config["path"])
    history_length = int(config["history_length"])
    forecast_horizon = int(config["forecast_horizon"])
    seasonality = int(config["seasonality"])
    features_per_window: List[List[float]] = []

    for values in series_collection:
        if len(values) < history_length + 3 * forecast_horizon:
            continue

        test_target_start = len(values) - forecast_horizon
        val_target_start = test_target_start - forecast_horizon
        train_last_end = val_target_start - forecast_horizon
        if train_last_end < history_length:
            continue

        candidate_end_indices = list(range(history_length, train_last_end + 1))
        if len(candidate_end_indices) > max_train_windows_per_series:
            step = (len(candidate_end_indices) - 1) / max(1, max_train_windows_per_series - 1)
            candidate_end_indices = sorted(
                {
                    candidate_end_indices[min(len(candidate_end_indices) - 1, int(round(step * idx)))]
                    for idx in range(max_train_windows_per_series)
                }
            )

        for end_index in candidate_end_indices:
            context = values[end_index - history_length : end_index]
            features_per_window.append(build_window_formation(context, seasonality=seasonality, end_index=end_index))

        val_context = values[val_target_start - history_length : val_target_start]
        test_context = values[test_target_start - history_length : test_target_start]
        features_per_window.append(build_window_formation(val_context, seasonality=seasonality, end_index=val_target_start))
        features_per_window.append(build_window_formation(test_context, seasonality=seasonality, end_index=test_target_start))

    columns = list(zip(*features_per_window))
    feature_stats: Dict[str, Dict[str, float]] = {}
    for feature_name, values in zip(FEATURE_ORDER, columns):
        numeric_values = [float(value) for value in values]
        feature_stats[feature_name] = {
            "mean": float(sum(numeric_values) / len(numeric_values)),
            "min": float(min(numeric_values)),
            "max": float(max(numeric_values)),
            **quantiles(numeric_values, [0.10, 0.25, 0.50, 0.75, 0.90]),
        }

    return {
        "dataset_name": config["name"],
        "series_count": len(series_collection),
        "sample_count": len(features_per_window),
        "history_length": history_length,
        "forecast_horizon": forecast_horizon,
        "seasonality": seasonality,
        "profile": config["profile"],
        "color": config["color"],
        "feature_stats": feature_stats,
    }


def format_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def rank_phrase(dataset_name: str, medians: Dict[str, float], absolute: bool = False) -> str:
    scored = {
        name: abs(value) if absolute else value
        for name, value in medians.items()
    }
    ordered = sorted(scored.items(), key=lambda item: item[1])
    names = [item[0] for item in ordered]
    if dataset_name == names[-1]:
        return "三者最高"
    if dataset_name == names[0]:
        return "三者最低"
    return "居中"


def slope_text(value: float, threshold: float = 0.03) -> str:
    if value > threshold:
        return "偏正，说明典型窗口在上行"
    if value < -threshold:
        return "偏负，说明典型窗口在下行"
    return "接近 0，说明典型窗口总体较平"


def compare_width_phrase(width: float, widths: Dict[str, float], dataset_name: str) -> str:
    label = rank_phrase(dataset_name, widths, absolute=False)
    if label == "三者最高":
        return f"分布最宽（q90-q10={format_number(width)}）"
    if label == "三者最低":
        return f"分布最窄（q90-q10={format_number(width)}）"
    return f"分布居中（q90-q10={format_number(width)}）"


def seasonal_strength_text(value: float) -> str:
    if value >= 0.7:
        return "季节模板复现很强"
    if value >= 0.45:
        return "季节性中等偏强"
    return "季节模板更容易漂移"


def regime_mix_text(value: float) -> str:
    if value >= 0.58:
        return "窗口里混合了较多状态成分"
    if value >= 0.45:
        return "状态混合度中等"
    return "窗口结构相对单纯"


def build_dataset_meaning(feature_name: str, dataset_name: str, dataset_stats: Dict[str, object], all_stats: Dict[str, Dict[str, object]]) -> str:
    current = dataset_stats["feature_stats"][feature_name]
    medians = {
        name: stats["feature_stats"][feature_name]["q50"]
        for name, stats in all_stats.items()
    }
    widths = {
        name: stats["feature_stats"][feature_name]["q90"] - stats["feature_stats"][feature_name]["q10"]
        for name, stats in all_stats.items()
    }
    median = current["q50"]
    width = current["q90"] - current["q10"]
    position = rank_phrase(dataset_name, medians)

    if feature_name == "local_slope":
        return f"中位数 {format_number(median)}，{slope_text(median)}；{compare_width_phrase(width, widths, dataset_name)}，说明短窗方向性 {'更强' if position != '三者最低' else '更弱'}。"
    if feature_name == "medium_slope":
        return f"中位数 {format_number(median)}，{slope_text(median, threshold=0.05)}；它比 local_slope 更像“最近半窗的方向”，在这个数据集上 {compare_width_phrase(width, widths, dataset_name)}。"
    if feature_name == "local_volatility":
        return "中位数固定为 1。这说明它主要承担归一化基准角色，本身不用于区分三个数据集的差异。"
    if feature_name == "volatility_ratio":
        if median > 1.05:
            state = "尾部比整窗更躁动"
        elif median < 0.95:
            state = "尾部比整窗更平静"
        else:
            state = "尾部与整窗波动接近"
        return f"中位数 {format_number(median)}，说明 {state}；{position}，可理解为最近阶段是否在“升温/降温”。"
    if feature_name == "seasonal_gap_norm":
        if median > 0.15:
            state = "当前相位通常高于上一个季节位置"
        elif median < -0.15:
            state = "当前相位通常低于上一个季节位置"
        else:
            state = "当前相位和上一个季节位置大体持平"
        return f"中位数 {format_number(median)}，说明 {state}；分布宽度 {format_number(width)}，反映季节对齐后仍有多大上下摆动。"
    if feature_name == "seasonal_strength":
        return f"中位数 {format_number(median)}，说明 {seasonal_strength_text(median)}；{position}，直接反映相邻两个 seasonal block 是否长得像。"
    if feature_name == "level_shift_norm":
        if median > 0.15:
            state = "窗口后段整体高于前段，存在抬升台阶"
        elif median < -0.15:
            state = "窗口后段整体低于前段，存在下沉台阶"
        else:
            state = "前后段均值接近，没有明显台阶"
        return f"中位数 {format_number(median)}，说明 {state}；{position}，表示这个数据集更常见还是更少见 level shift。"
    if feature_name == "max_zscore":
        return f"中位数 {format_number(median)}，{position}；数值越高越容易在窗口里看到尖峰或深坑，这个数据集的极端点强度 {('更突出' if position == '三者最高' else '更温和' if position == '三者最低' else '居中')}。"
    if feature_name == "range_norm":
        return f"中位数 {format_number(median)}，{position}；表示典型窗口的峰谷振幅 {'更大' if position == '三者最高' else '更小' if position == '三者最低' else '中等'}。"
    if feature_name == "phase_sin":
        return f"q10={format_number(current['q10'])}，q90={format_number(current['q90'])}，说明样本覆盖了大量季相位置。它的意义不是强弱，而是告诉模型“当前终点在周期圆上的哪一边”。"
    if feature_name == "phase_cos":
        return f"q10={format_number(current['q10'])}，q90={format_number(current['q90'])}，同样是相位坐标的一部分。它和 phase_sin 一起让月份/星期的首尾位置连续。"
    if feature_name == "change_proxy":
        if median >= 0.5:
            state = "典型窗口里常有较明显的单步跳变"
        elif median <= 0.2:
            state = "单步跳变通常不大"
        else:
            state = "跳变幅度中等"
        return f"中位数 {format_number(median)}，说明 {state}；{position}，可视作这个数据集局部突发变化的敏感度。"
    if feature_name == "curvature":
        if median > 0.02:
            state = "窗口更常见向上拱/加速上行"
        elif median < -0.02:
            state = "窗口更常见向下拱/加速下行"
        else:
            state = "整体更接近线性变化"
        return f"中位数 {format_number(median)}，说明 {state}；宽度 {format_number(width)} 决定拐弯程度在不同窗口间变化多不多。"
    if feature_name == "autocorr_lag1":
        if median >= 0.9:
            state = "相邻点极度连续，曲线很平滑"
        elif median >= 0.4:
            state = "相邻点具有明显延续性"
        else:
            state = "相邻点延续性较弱，更容易拐或跳"
        return f"中位数 {format_number(median)}，说明 {state}；{position}，是三个数据集里时间连续性的直接刻画。"
    if feature_name == "stability_score":
        return f"中位数 {format_number(median)}，{position}；分数越高越稳，所以这个数据集的典型窗口 {'更稳定' if position == '三者最高' else '更不稳定' if position == '三者最低' else '稳定性中等'}。"
    if feature_name == "regime_mix_score":
        return f"中位数 {format_number(median)}，说明 {regime_mix_text(median)}；{position}，反映单个窗口里趋势、波动和台阶是否一起出现。"
    return ""


def svg_distribution(feature_name: str, all_stats: Dict[str, Dict[str, object]]) -> str:
    domain_min = min(stats["feature_stats"][feature_name]["min"] for stats in all_stats.values())
    domain_max = max(stats["feature_stats"][feature_name]["max"] for stats in all_stats.values())
    if abs(domain_max - domain_min) < 1e-8:
        domain_min -= 1.0
        domain_max += 1.0

    width = 520
    height = 144
    left = 120
    right = 28
    top = 16
    row_gap = 38
    inner_width = width - left - right

    def scale_x(value: float) -> float:
        return left + (value - domain_min) / (domain_max - domain_min) * inner_width

    lines = [
        f'<svg viewBox="0 0 {width} {height}" class="dist-svg" role="img" aria-label="{escape(feature_name)} distribution">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fcfcfb" rx="16"/>',
    ]

    if domain_min < 0.0 < domain_max:
        zero_x = scale_x(0.0)
        lines.append(f'<line x1="{zero_x:.1f}" y1="8" x2="{zero_x:.1f}" y2="{height - 12}" stroke="#d1d5db" stroke-dasharray="4 4"/>')

    for row_index, config in enumerate(DATASET_CONFIGS):
        stats = all_stats[config["name"]]["feature_stats"][feature_name]
        y = top + row_index * row_gap + 12
        q10 = scale_x(stats["q10"])
        q25 = scale_x(stats["q25"])
        q50 = scale_x(stats["q50"])
        q75 = scale_x(stats["q75"])
        q90 = scale_x(stats["q90"])
        minimum = scale_x(stats["min"])
        maximum = scale_x(stats["max"])
        color = config["color"]

        lines.append(f'<text x="18" y="{y + 5:.1f}" fill="#111827" font-size="13" font-family="Segoe UI, PingFang SC, Microsoft YaHei, sans-serif">{escape(config["name"])}</text>')
        lines.append(f'<line x1="{minimum:.1f}" y1="{y:.1f}" x2="{maximum:.1f}" y2="{y:.1f}" stroke="#cbd5e1" stroke-width="2"/>')
        lines.append(f'<line x1="{q10:.1f}" y1="{y:.1f}" x2="{q90:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="4" opacity="0.45"/>')
        lines.append(f'<line x1="{q25:.1f}" y1="{y:.1f}" x2="{q75:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="10" stroke-linecap="round"/>')
        lines.append(f'<circle cx="{q50:.1f}" cy="{y:.1f}" r="6" fill="#ffffff" stroke="{color}" stroke-width="3"/>')
    lines.append(
        f'<text x="{left}" y="{height - 10}" fill="#6b7280" font-size="12" font-family="Segoe UI, PingFang SC, Microsoft YaHei, sans-serif">domain [{format_number(domain_min)}, {format_number(domain_max)}]</text>'
    )
    lines.append("</svg>")
    return "".join(lines)


def feature_card_html(feature_name: str, all_stats: Dict[str, Dict[str, object]]) -> str:
    spec = FEATURE_SPECS[feature_name]
    stat_rows = []
    meaning_rows = []
    for config in DATASET_CONFIGS:
        dataset_stats = all_stats[config["name"]]
        stats = dataset_stats["feature_stats"][feature_name]
        stat_rows.append(
            "<tr>"
            f"<td>{escape(config['name'])}</td>"
            f"<td>{format_number(stats['q50'])}</td>"
            f"<td>{format_number(stats['q25'])} ~ {format_number(stats['q75'])}</td>"
            f"<td>{format_number(stats['q10'])} ~ {format_number(stats['q90'])}</td>"
            "</tr>"
        )
        meaning_rows.append(
            f"<div class='meaning-row'><strong>{escape(config['name'])}</strong><p>{escape(build_dataset_meaning(feature_name, config['name'], dataset_stats, all_stats))}</p></div>"
        )

    return (
        "<section class='feature-card'>"
        f"<h2>{escape(spec['title'])}</h2>"
        "<div class='feature-grid'>"
        "<div>"
        "<div class='meta-label'>计算公式</div>"
        f"<pre>{escape(spec['formula'])}</pre>"
        "<div class='meta-label'>直觉解释</div>"
        f"<p class='intuition'>{escape(spec['intuition'])}</p>"
        "</div>"
        f"<div>{svg_distribution(feature_name, all_stats)}</div>"
        "</div>"
        "<div class='table-wrap'>"
        "<table>"
        "<thead><tr><th>数据集</th><th>中位数</th><th>IQR</th><th>q10 ~ q90</th></tr></thead>"
        f"<tbody>{''.join(stat_rows)}</tbody>"
        "</table>"
        "</div>"
        "<div class='meta-label'>在三类数据集上各自意味着什么</div>"
        f"{''.join(meaning_rows)}"
        "</section>"
    )


def build_html(report_payload: Dict[str, object]) -> str:
    dataset_summary_cards = []
    for config in DATASET_CONFIGS:
        stats = report_payload["datasets"][config["name"]]
        dataset_summary_cards.append(
            "<div class='dataset-card'>"
            f"<h3>{escape(config['name'])}</h3>"
            f"<p>{escape(stats['profile'])}</p>"
            "<ul>"
            f"<li>series_count = {stats['series_count']}</li>"
            f"<li>sample_count = {stats['sample_count']}</li>"
            f"<li>history_length = {stats['history_length']}</li>"
            f"<li>forecast_horizon = {stats['forecast_horizon']}</li>"
            f"<li>seasonality = {stats['seasonality']}</li>"
            "</ul>"
            "</div>"
        )

    feature_cards = [feature_card_html(feature_name, report_payload["datasets"]) for feature_name in FEATURE_ORDER]
    generated_at = escape(report_payload["generated_at"])

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Formation 特征图解</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --card: #fffdf8;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #e5e7eb;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 30%),
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 34%),
        linear-gradient(180deg, #faf6ef 0%, var(--bg) 100%);
      line-height: 1.6;
    }}
    .page {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 36px 24px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,250,240,0.86));
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 28px;
      padding: 30px 32px;
      box-shadow: 0 18px 60px rgba(15, 23, 42, 0.08);
      margin-bottom: 24px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 34px;
      line-height: 1.15;
    }}
    .hero p {{
      margin: 8px 0;
      color: #374151;
      max-width: 1000px;
    }}
    .stamp {{
      color: var(--muted);
      font-size: 14px;
    }}
    .datasets {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 26px;
    }}
    .dataset-card, .feature-card {{
      background: var(--card);
      border: 1px solid rgba(15, 23, 42, 0.08);
      border-radius: 24px;
      box-shadow: 0 16px 40px rgba(15, 23, 42, 0.07);
    }}
    .dataset-card {{
      padding: 20px 22px;
    }}
    .dataset-card h3 {{
      margin: 0 0 8px;
      font-size: 20px;
    }}
    .dataset-card p {{
      margin: 0 0 10px;
      color: #4b5563;
    }}
    .dataset-card ul {{
      margin: 0;
      padding-left: 18px;
      color: #374151;
    }}
    .feature-card {{
      padding: 22px;
      margin-bottom: 18px;
    }}
    .feature-card h2 {{
      margin: 0 0 16px;
      font-size: 24px;
    }}
    .feature-grid {{
      display: grid;
      grid-template-columns: minmax(320px, 1fr) minmax(400px, 540px);
      gap: 18px;
      align-items: start;
    }}
    .meta-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 6px;
      font-weight: 700;
    }}
    pre {{
      margin: 0 0 14px;
      padding: 14px 16px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 13px;
    }}
    .intuition {{
      margin: 0;
      color: #374151;
    }}
    .dist-svg {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 16px;
    }}
    .table-wrap {{
      overflow-x: auto;
      margin: 16px 0 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-weight: 700;
      background: #fafaf9;
    }}
    .meaning-row {{
      padding: 12px 14px;
      background: #fafaf8;
      border: 1px solid var(--line);
      border-radius: 14px;
      margin-top: 10px;
    }}
    .meaning-row strong {{
      display: inline-block;
      margin-bottom: 4px;
    }}
    .meaning-row p {{
      margin: 0;
      color: #374151;
    }}
    @media (max-width: 980px) {{
      .feature-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <h1>16 个 formation 特征图解</h1>
      <p>每个特征都按同一格式展开：<strong>计算公式 + 直觉解释 + 在三类 forecasting 数据集上各自意味着什么</strong>。图里的每一行是一个数据集：细线表示 min-max，彩色淡线表示 q10-q90，粗线表示 q25-q75，白心圆点表示中位数。</p>
      <p>变量约定：<code>x_1...x_T</code> 为当前历史窗口，<code>scale</code> 是窗口内的稳健尺度，<code>s</code> 是 seasonality，<code>end_index</code> 是窗口终点在全序列中的位置。</p>
      <p class="stamp">生成时间：{generated_at}</p>
    </section>
    <section class="datasets">
      {''.join(dataset_summary_cards)}
    </section>
    {''.join(feature_cards)}
  </main>
</body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {
        config["name"]: collect_dataset_feature_stats(config)
        for config in DATASET_CONFIGS
    }
    report_payload = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "datasets": datasets,
        "feature_specs": FEATURE_SPECS,
    }

    STATS_PATH.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    HTML_PATH.write_text(build_html(report_payload), encoding="utf-8")

    print(f"wrote {STATS_PATH}")
    print(f"wrote {HTML_PATH}")


if __name__ == "__main__":
    main()
