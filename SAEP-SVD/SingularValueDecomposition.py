import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

def compress_lora_svd(input_path: str, output_path: str, target_rank: int, device: str = "cuda"):
    # 1. Считываем метаданные
    with safe_open(input_path, framework="pt") as f:
        metadata = f.metadata() or {}

    state_dict = load_file(input_path)
    new_state_dict = {}
    pairs = {}

    # 2. Правильно группируем пары весовых матриц A и B
    for key in state_dict.keys():
        if ".lora_down.weight" in key or ".lora_A.weight" in key:
            base = key.replace(".lora_down.weight", "").replace(".lora_A.weight", "")
            pairs.setdefault(base, {})["A"] = key
        elif ".lora_B.weight" in key or ".lora_up.weight" in key:
            base = key.replace(".lora_up.weight", "").replace(".lora_B.weight", "")
            pairs.setdefault(base, {})["B"] = key
        else:
            new_state_dict[key] = state_dict[key]

    print(f"Найдено {len(pairs)} валидных пар матриц для SVD-разложения.")
    print(f"Запуск SVD до ранга {target_rank} на {device}...")

    processed_count = 0

    # 3. Пересчитываем каждый слой через SVD с прогресс-баром Rich
    with Progress(
        SpinnerColumn("dots", style="bold yellow"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=35, style="black", complete_style="bold green", finished_style="bold green"),
        TaskProgressColumn("[bold yellow]{task.percentage:>3.0f}%"),
        TextColumn("• [bold magenta]{task.completed}/{task.total}[/] пар"),
        TextColumn("• Прошло:"),
        TimeElapsedColumn(),
        TextColumn("• Осталось:"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"SVD (до {target_rank})", total=len(pairs))

        for base_key, pair in pairs.items():
            if "A" not in pair or "B" not in pair:
                print(f"Предупреждение: Пропущена неполная пара для {base_key}")
                progress.update(task, advance=1)
                continue

            key_A, key_B = pair["A"], pair["B"]
            w_A = state_dict[key_A].to(device, dtype=torch.float32)
            w_B = state_dict[key_B].to(device, dtype=torch.float32)

            # Поддержка 2D и 4D (Conv2d) тензоров
            orig_shape_A = w_A.shape
            orig_shape_B = w_B.shape
            if w_A.dim() == 4:
                w_A = w_A.squeeze(-1).squeeze(-1)
                w_B = w_B.squeeze(-1).squeeze(-1)

            # Восстанавливаем дельту весов W = B @ A
            delta_w = torch.matmul(w_B, w_A)

            # SVD-разложение
            U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)

            # Отсекаем шум под target_rank
            r = min(target_rank, S.shape[0])
            U_r = U[:, :r]
            S_r = S[:r]
            Vh_r = Vh[:r, :]

            # Распределяем энергию сингулярных значений
            sqrt_S = torch.diag(torch.sqrt(S_r))
            new_B = torch.matmul(U_r, sqrt_S)
            new_A = torch.matmul(sqrt_S, Vh_r)

            # Возвращаем исходную размерность, если была Conv2d
            if len(orig_shape_A) == 4:
                new_A = new_A.unsqueeze(-1).unsqueeze(-1)
                new_B = new_B.unsqueeze(-1).unsqueeze(-1)

            orig_dtype = state_dict[key_A].dtype
            new_state_dict[key_A] = new_A.to(orig_dtype).cpu()
            new_state_dict[key_B] = new_B.to(orig_dtype).cpu()
            
            # Обновляем alpha под новый ранг, если скаляр существует
            alpha_key = base_key + ".alpha"
            if alpha_key in new_state_dict:
                new_state_dict[alpha_key] = torch.tensor(float(r), dtype=torch.float32)

            processed_count += 1
            progress.update(task, advance=1)

    print(f"Успешно обработано слоев: {processed_count}")

    # 4. Обновляем метаданные и сохраняем
    new_metadata = dict(metadata)
    new_metadata["ss_network_dim"] = str(target_rank)
    new_metadata["ss_network_alpha"] = str(target_rank)

    save_file(new_state_dict, output_path, metadata=new_metadata)
    print(f"Готово! Полноценная модель сохранена в {output_path}")

if __name__ == "__main__":
    compress_lora_svd(
        input_path="Merg_VanGogh_Aivazovsky_Flux2.safetensors",
        output_path="VanGogh_MERGE_Aivazovsky.safetensors",
        target_rank=128
    )