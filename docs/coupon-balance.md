# 代金券余额：为什么不能直接用模型密钥查

一句话：**模型密钥能证明"这次推理算谁的账"，但不能读账。** 查余额是另一个产品、另一套凭据。

| | 模型调用 | 查代金券余额 |
|---|---|---|
| 凭据 | 百炼 API Key（`sk-...`） | 阿里云 AccessKey（AK/SK） |
| 产品 | 百炼 Model Studio | 费用与成本 BssOpenApi |
| 权限 | 调模型 | RAM 权限点 `bss:DescribeCoupons` |
| 端点 | `dashscope.aliyuncs.com` | `business.aliyuncs.com` |

拿模型密钥去调账单接口，只会得到一条看不懂的 401。所以界面上那块余额的三种状态是**分开**的：

| 状态 | 界面 | 含义 |
|---|---|---|
| `configured=false` | 一段"怎么配"的说明 + 重试按钮 | 还没配 AK。**不是错误**，问答照常可用 |
| `ok=false` | 红色块：上游 `Code` + `Message` + 一句提示 | 配了 AK 但被拒绝，原样转述官方错误 |
| `ok=true` | 大号数字 + 每张券明细 | 查到了 |

## 配法（5 分钟）

1. 控制台 → 访问控制 RAM → 用户 → 创建用户，勾选 **OpenAPI 调用访问**
2. 授权：挂 `AliyunBSSReadOnlyAccess`（只读，够用；不要给管理员）
3. 把 AccessKey ID / Secret 填进 `.env`：

```
ALIYUN_ACCESS_KEY_ID=LTAI...
ALIYUN_ACCESS_KEY_SECRET=...
```

`.env` 已被 `.gitignore` 拦住，不会进仓库。

## 查的是哪个接口，为什么不是另一个

`QueryCashCoupons`（BssOpenApi 2017-12-14）。返回每张券的 `NominalValue`（面额）、`Balance`（余额）、`Status`、`ExpiryTime`、`ApplicableProducts`。

**不能只调 `QueryAccountBalance`**：官方文档明确它返回的是"可用额度/现金余额"，**不含代金券**。只调它会出现"界面显示 ¥0，控制台明明有 300 券"的假象——这比不显示更糟。

合计只加 `Status == "Available"` 的券；过期/作废的也列出来，方便对照。

## 验证到哪一步（重要）

2026-11-18 用一对**假 AK** 打过 `QueryAccountBalance` 与 `QueryCashCoupons`：返回 `InvalidAccessKeyId.NotFound`，**不是** `SignatureDoesNotMatch`。这说明端点、Action 名、RPC 签名格式都被服务端认了，剩下的只是密钥真不真。

**没有用真实 AK 跑通过**（手上没有）。所以第一次配置时如果失败，错误块里给的是阿里云自己那句话，不是我的猜测。常见几条：

| 上游 Code | 意思 | 怎么办 |
|---|---|---|
| `InvalidAccessKeyId.NotFound` | AK 不存在 | 检查有没有多空格、是不是别的账号的 |
| `SignatureDoesNotMatch` | AK/SK 不配套 | 多半是 SK 少复制了几位 |
| `NoPermission` / `NotAuthorized` | RAM 用户没权限 | 挂 `AliyunBSSReadOnlyAccess` |
| `NotApplicable` | 这种账号类型不能这样调 | 换主账号（root）的 AK |

## 已知限制

- **30 秒缓存**：余额变化是"分"级的，连点刷新不会把上游打爆；按钮上的刷新会带 `?refresh=true` 绕过缓存。
- **国际站要显式填端点**：`ALIYUN_BSS_ENDPOINT=https://business.ap-southeast-1.aliyuncs.com`。国际站账号打国内站端点通常表现为**超时**，而不是明确报错——所以这个参数是必须的，不是可选的优化。
- **只显示代金券**，不显示现金余额。要一起看就再加一次 `QueryAccountBalance` 调用，但那就多一个权限点（`bss:DescribeAccount`）；目前这张 300 元的券是主要余额，先只做券。
- **不记录历史曲线**：只显示"现在还剩多少"。要看每天花多少，去费用与成本控制台。
