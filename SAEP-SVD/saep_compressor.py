import os
import time
import torch
from safetensors.torch import load_file, save_file

# Попытка подключить красивый прогресс-бар tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def saep_svd_compress_tensor(w_full: torch.Tensor, target_rank: int = 512):
    dtype_orig = w_full.dtype
    w_float = w_full.to(DEVICE, dtype=torch.float32)
    
    norm_orig = torch.norm(w_float, p='fro')
    if norm_orig == 0:
        return w_full.to(dtype_orig)
        
    U, S, Vh = torch.linalg.svd(w_float, full_matrices=False)
    
    r = min(target_rank, S.shape[0])
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    
    # Sign Alignment (GPU)
    proj = torch.sum(U_r * torch.matmul(w_float, Vh_r.T), dim=0)
    signs = torch.sign(proj)
    signs[signs == 0] = 1.0
    U_r = U_r * signs.unsqueeze(0)
            
    # Energy Preservation
    norm_distilled = torch.norm(S_r, p=2)
    gamma = norm_orig / (norm_distilled + 1e-8)
    S_r_scaled = S_r * gamma
    
    W_reconstructed = torch.matmul(U_r, torch.matmul(torch.diag(S_r_scaled), Vh_r))
    return W_reconstructed.to(dtype_orig)

def find_lora_pairs(state_dict):
    pairs = []
    keys = list(state_dict.keys())
    
    for k in keys:
        if ".lora_down.weight" in k:
            up_k = k.replace(".lora_down.weight", ".lora_up.weight")
            if up_k in state_dict:
                pairs.append((k, up_k))
        elif ".lora_down" in k and not k.endswith(".weight"):
            up_k = k.replace(".lora_down", ".lora_up")
            if up_k in state_dict:
                pairs.append((k, up_k))
        elif ".lora_A.weight" in k:
            up_k = k.replace(".lora_A.weight", ".lora_B.weight")
            if up_k in state_dict:
                pairs.append((k, up_k))
        elif ".lora_A" in k and not k.endswith(".weight"):
            up_k = k.replace(".lora_A", ".lora_B")
            if up_k in state_dict:
                pairs.append((k, up_k))
        elif ".down.weight" in k:
            up_k = k.replace(".down.weight", ".up.weight")
            if up_k in state_dict:
                pairs.append((k, up_k))

    return pairs

def process_lora_saep(input_path: str, output_path: str, target_rank: int = 512):
    print(f"[+] Вычисления на: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[+] Ускоритель: {torch.cuda.get_device_name(0)}")
        
    print(f"[+] Загрузка файла: {input_path}")
    state_dict = load_file(input_path)
    
    pairs = find_lora_pairs(state_dict)
    total_pairs = len(pairs)
    print(f"[+] Найдено {total_pairs} пар слоев LoRA.")
    
    if total_pairs == 0:
        print("[!] Ошибка: Слои LoRA не найдены!")
        return

    new_state_dict = dict(state_dict)
    start_time = time.time()
    
    print("\n[🚀] Запуск SAEP-SVD компрессии...\n")
    
    iterator = tqdm(pairs, desc="Обработка слоев", unit="слой") if HAS_TQDM else pairs

    for idx, (down_key, up_key) in enumerate(iterator, 1):
        w_down = state_dict[down_key].to(DEVICE, dtype=torch.float32)
        w_up = state_dict[up_key].to(DEVICE, dtype=torch.float32)
        
        if len(w_up.shape) == 4:
            w_full = torch.matmul(w_up.squeeze(), w_down.squeeze())
        else:
            w_full = torch.matmul(w_up, w_down)
            
        w_compressed = saep_svd_compress_tensor(w_full, target_rank=target_rank)
        
        U, S, Vh = torch.linalg.svd(w_compressed.to(DEVICE, dtype=torch.float32), full_matrices=False)
        r = min(target_rank, S.shape[0])
        sqrt_S = torch.diag(torch.sqrt(S[:r]))
        
        new_up = torch.matmul(U[:, :r], sqrt_S).to(state_dict[up_key].dtype).cpu()
        new_down = torch.matmul(sqrt_S, Vh[:r, :]).to(state_dict[down_key].dtype).cpu()
        
        if len(state_dict[up_key].shape) == 4:
            new_up = new_up.unsqueeze(-1).unsqueeze(-1)
            new_down = new_down.unsqueeze(-1).unsqueeze(-1)
            
        new_state_dict[up_key] = new_up
        new_state_dict[down_key] = new_down
        
        # Запасной вывод прогресса, если нет tqdm
        if not HAS_TQDM and (idx % 5 == 0 or idx == total_pairs):
            elapsed = time.time() - start_time
            percent = (idx / total_pairs) * 100
            speed = idx / elapsed if elapsed > 0 else 0
            eta = (total_pairs - idx) / speed if speed > 0 else 0
            layer_short = down_key.split('.')[-3] if len(down_key.split('.')) > 2 else down_key
            print(f"[{percent:5.1f}%] {idx}/{total_pairs} | {speed:.1f} слоев/сек | Осталось: {eta:.0f}с | Слой: {layer_short}")

    total_time = time.time() - start_time
    print(f"\n[+] Сжатие завершено за {total_time:.2f} сек!")
    print(f"[+] Сохранение итогового файла: {output_path}")
    
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass
            
    save_file(new_state_dict, output_path)
    print(f"\n[✓] УСПЕШНО СОХРАНЕНО: {output_path}")

if __name__ == "__main__":
    INPUT_LORA = "Merg_VanGogh_Aivazovsky_Flux2.safetensors"
    OUTPUT_LORA = "R256-VanGogh-Aivazovsky-SAEP-SVD.safetensors"
    
    if os.path.exists(INPUT_LORA):
        process_lora_saep(INPUT_LORA, OUTPUT_LORA, target_rank=256)
    else:
        print(f"[!] ОШИБКА: Файл {INPUT_LORA} не найден в текущей папке!")