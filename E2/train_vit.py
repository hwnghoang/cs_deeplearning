import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ==========================================
# 1. BƯỚC 1: TỰ VIẾT MULTI-HEAD SELF-ATTENTION (MSA)
# ==========================================
class FromScratchMSA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim phải chia hết cho num_heads"
        
        # Các trọng số tuyến tính cho Q, K, V và đầu ra
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        B, N, C = x.shape
        
        # Tạo Q, K, V và chia thành nhiều heads: (B, num_heads, N, head_dim)
        q = self.q_linear(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Tính Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Nhân với V và ghép các head lại
        out = torch.matmul(attn_weights, v) # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, C) # (B, N, embed_dim)
        
        return self.out_linear(out)

# ==========================================
# 2. BƯỚC 2: TOKENIZATION (PATCH EMBEDDING)
# ==========================================
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Dùng Conv2d để cắt ảnh thành các patch và chiếu lên không gian embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x shape: (B, 1, 28, 28)
        x = self.proj(x) # (B, embed_dim, 4, 4 nếu patch_size=7)
        x = x.flatten(2) # (B, embed_dim, 16)
        x = x.transpose(1, 2) # (B, 16, embed_dim)
        return x

# ==========================================
# 3. BƯỚC 3: XÂY DỰNG MÔ HÌNH VISION TRANSFORMER (ViT)
# ==========================================
class SimpleViT(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_channels=1, num_classes=10, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Thêm Class Token và Positional Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.patch_embed.n_patches, embed_dim))
        
        # Khối Transformer Encoder đơn giản (dùng MSA tự viết ở trên)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(embed_dim),
                'msa': FromScratchMSA(embed_dim, num_heads),
                'norm2': nn.LayerNorm(embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(embed_dim * 2, embed_dim)
                )
            }) for _ in range(num_layers)
        ])
        
        # Classification Head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x) # (B, N, embed_dim)
        
        # Nối [CLS] token vào chuỗi token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Cộng Positional Encoding
        x = x + self.pos_embed
        
        # Cho qua các Transformer layers
        for layer in self.layers:
            x = x + layer['msa'](layer['norm1'](x))
            x = x + layer['ffn'](layer['norm2'](x))
            
        # Lấy đại diện từ [CLS] token để phân loại
        cls_out = x[:, 0]
        return self.head(cls_out)

# ==========================================
# 4. BƯỚC 4: CHUẨN BỊ DỮ LIỆU & HUẤN LUYỆN
# ==========================================
if __name__ == '__main__':
    # Thiết bị huấn luyện (GPU nếu có, không thì CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # Tải tập Fashion-MNIST qua torchvision
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    # Khởi tạo mô hình, hàm loss và optimizer
    model = SimpleViT(img_size=28, patch_size=7, embed_dim=64, num_heads=4, num_layers=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    # Vòng lặp huấn luyện thủ công chuẩn PyTorch (Forward -> Loss -> Backward -> Step)[cite: 4]
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()         # 1. Xóa gradient cũ
            outputs = model(inputs)       # 2. Lan truyền tiến (Forward)
            loss = criterion(outputs, labels) # 3. Tính hàm mất mát
            loss.backward()               # 4. Lan truyền ngược (Backward)
            optimizer.step()              # 5. Cập nhật trọng số
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss trung bình: {running_loss/len(trainloader):.4f}")

    print("Huấn luyện mô hình E2 (ViT với MSA tự viết) hoàn tất!")