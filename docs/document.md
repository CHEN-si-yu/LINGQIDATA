# 灵启数据 API 接口文档

本文档详细介绍了灵启数据提供的各类 API 接口，包括股票、港股、可转债、ETF、指数及实时行情等数据服务。

## 1. 概览 (Overview)

### 1.1 基础地址 (Base URL)
`https://data.diemeng.chat/api`

### 1.2 鉴权方式 (Authentication)
所有接口都需要进行鉴权。请在请求头 (Header) 中携带 `apiKey`。
- **Header Key**: `apiKey` (推荐) 或 `X-API-Key`
- **Value**: 您的 API 密钥 (可在个人中心获取)

**示例 (Curl)**:
```bash
curl -H "apiKey: your_api_key" https://data.diemeng.chat/api/stock/list
```

### 1.3 通用返回参数
所有接口均返回 JSON 格式数据，包含以下通用字段：
- `code` (int): 状态码。200 表示成功，其他表示失败。
- `msg` (string): 提示信息。成功时为 "Success"，失败时为错误描述。
- `data` (object/array): 具体的业务数据。

### 1.4 常见错误码
- `401`: 未授权 (Missing or invalid apiKey)
- `403`: 权限不足 (Insufficient permissions)
- `429`: 请求过于频繁 (Rate limit exceeded)
- `500`: 服务器内部错误

---

## 2. 接口列表 (Interface List)

| 分类 | 接口名称 | 方法 | 路径 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| **基础数据** | 获取交易日历 | GET/POST | `/api/basic/calendar` | 获取交易日历数据 |
| **港股** | 获取港股列表 | GET | `/api/stock/hk/list` | 获取所有港股基础信息列表 |
| | 获取互通数据 | GET | `/api/stock/hk/connect` | 获取港深上三个市场的互通情况 |
| | 获取港股财务数据 | POST | `/api/stock/hk/finance` | 获取港股财务数据 |
| **可转债** | 获取分时数据 | POST | `/api/bond/history` | 获取可转债的历史分时交易数据 |
| | 获取日K线数据 | POST | `/api/bond/daily` | 获取可转债的日K线历史数据 |
| | 获取日指标数据 | POST | `/api/bond/indicator_daily` | 获取可转债的日指标数据 |
| | 收盘快照 | POST | `/api/bond/closing_snapshot` | 获取可转债收盘时的快照数据 |
| | 获取可转债列表 | POST | `/api/bond/list` | 获取可转债基本信息列表 |
| **ETF** | 获取实时分时数据 | POST | `/api/etf/realtime/history` | 获取ETF实时1分钟级别分时数据(支持最近7天内) |
| | 获取分时数据 | POST | `/api/etf/history` | 获取ETF的历史分时交易数据 |
| | 获取日K线数据 | POST | `/api/etf/daily` | 获取ETF的日K线历史数据 |
| | 获取日K线数据(前复权) | POST | `/api/etf/daily_adj` | 获取指定ETF的前复权日K线数据 |
| | 获取分钟K线数据(复权) | POST | `/api/etf/min_adj` | 获取指定ETF的复权分钟K线数据 |
| | 获取复权因子 | POST | `/api/etf/adj_factor` | 获取指定ETF的复权因子数据 |
| | 复权因子变更查询 | POST | `/api/etf/adj_factor/changes` | 查询指定日期是否有复权因子变更 |
| **股票** | 获取每日财务数据 | POST | `/api/stock/finance` | 获取指定股票的每日财务指标数据 |
| | 获取财务指标报表数据 | POST | `/api/stock/financial_indicator` | 从 stock_financial_indicator 表获取财务指标数据 |
| | 获取利润表数据 | POST | `/api/stock/income` | 从 stock_income 表获取利润表数据 |
| | 获取资产负债表数据 | POST | `/api/stock/balancesheet` | 从 stock_balancesheet 表获取资产负债表数据 |
| | 获取现金流量表数据 | POST | `/api/stock/cashflow` | 从 stock_cashflow 表获取现金流量表数据 |
| | 获取大小单资金金流向 | POST | `/api/stock/main_fund_flow` | 获取大小单资金金流向数据 |
| | 获取主力资金流向总览 | POST | `/api/stock/main_fund_flow_overview` | 获取主力资金流向总览数据 |
| | 获取筹码峰分布 | POST | `/api/stock/cyq_chips` | 获取股票筹码峰分布数据 |
| | 获取涨停数据 | POST | `/api/stock/limit_up` | 获取涨停明细数据 |
| | 获取涨跌停数据 | POST | `/api/stock/limit_list` | 获取涨跌停数据 |
| | 获取ST信息 | POST | `/api/stock/st_info` | 获取ST股票信息 |
| | 获取股票列表 | GET | `/api/stock/list` | 获取所有股票的基础信息列表 |
| | 获取日K线数据 | POST | `/api/stock/daily` | 获取指定股票的日线级别行情数据 |
| | 获取日K线数据(复权) | POST | `/api/stock/daily_adj` | 获取指定股票的复权日K线数据 |
| | 获取周期K线数据 | POST | `/api/stock/kline` | 获取指定股票的周期K线(周/月)数据 |
| | 获取周期K线数据(复权) | POST | `/api/stock/kline_adj` | 获取指定股票的复权周期K线(周/月)数据 |
| | 获取分钟K线数据(复权) | POST | `/api/stock/min_adj` | 获取指定股票的分钟级复权K线数据 |
| | 获取复权因子(涨跌幅算法) | POST | `/api/stock/adj_factor` | 获取指定股票的复权因子数据 |
| | 复权因子变更查询 | POST | `/api/stock/adj_factor/changes` | 查询指定日期是否有复权因子变更 |
| | 获取历史分时 | POST | `/api/stock/history` | 获取指定股票的历史分时数据 |
| | 下载全市场当天全部分时 | POST | `/api/stock/daily_dump` | 下载全市场当天的全部数据 |
| | 获取股票停牌信息 | GET | `/api/stock/suspension` | 获取股票停牌信息 |
| | 竞价结果快照 | POST | `/api/stock/call_auction` | 获取集合竞价时段的数据 |
| | 收盘快照 | POST | `/api/stock/closing_snapshot` | 获取收盘时的快照数据 |
| | 每日快照数据 | POST | `/api/stock/snapshot_daily` | 实时更新的股票快照数据 |
| | Websocket推送历史 | POST | `/api/stock/snapshot_push_history` | 获取当天每一次 WebSocket 推送的历史记录 |
| | 股票条件搜索 | POST | `/api/stock/search` | 通过自然语言搜索语句查询符合条件的股票 |
| | 获取券商盈利预测 | POST | `/api/stock/report_rc` | 获取券商每天研报的盈利预测数据（stock_code/report_date/时间区间三选一必传） |
| | 获取龙虎榜机构明细 | POST | `/api/stock/dragon_tiger` | 获取龙虎榜机构明细数据 |
| | 获取龙虎榜每日明细 | POST | `/api/stock/top_list` | 获取龙虎榜每日交易明细数据 |
| **指数** | 获取TDX板块列表 | GET | `/api/tdx/blocks` | 获取通达信板块列表数据 |
| | 获取TDX板块成分股 | GET | `/api/tdx/block_stocks` | 获取通达信板块成分股数据 |
| | 获取TDX板块日K | GET | `/api/tdx/daily` | 获取通达信板块指数日K线数据 |
| | 获取东财板块列表 | POST | `/api/dc/blocks` | 获取东方财富板块列表数据 |
| | 获取东财板块成分股 | POST | `/api/dc/block_stocks` | 获取东方财富板块成分股数据 |
| | 获取东财板块日K | POST | `/api/dc/daily` | 获取东方财富板块指数日K线数据 |
| | 获取同花顺板块分类 | POST | `/api/index/ths_sector_categories` | 获取同花顺板块分类数据 |
| | 获取同花顺成分股 | POST | `/api/index/ths_constituent_stocks` | 获取同花顺成分股数据 |
| | 获取同花顺日线数据 | POST | `/api/index/ths_daily` | 获取同花顺指数日K线数据 |
| | 获取历史分时 | POST | `/api/index/history` | 获取指定指数的历史分时数据 |
| | 获取指数成分和权重 | POST | `/api/index/weight` | 获取指数月度成分和权重数据 |
| **期货** | 获取合约基础信息 | POST | `/api/future/basic` | 获取期货合约的基础信息数据 |
| | 获取主连合约映射 | POST | `/api/future/mapping` | 获取期货主连或连续合约与实际月合约的映射关系 |
| | 获取分钟K线数据 | POST | `/api/future/minute` | 获取期货合约的历史分钟K线数据（接口最多返回10000条） |
| **实时行情** | 竞价快照数据 | POST | `/api/realtime/auction_daily` | 获取当天竞价时间(09:15-09:25)的快照数据 |
| | WebSocket 快照推送 | WS | `wss://data.diemeng.chat/ws/stock/snapshot` | 实时推送全市场股票的快照数据 |
| | 涨跌停状态 WS 推送 | WS | `wss://data.diemeng.chat/ws/stream` | 订阅涨跌停状态变化与持续更新事件 |

---

## 3. 接口详情 (Interface Details)

### 3.1 基础数据 (Basic Data)

#### 3.1.1 获取交易日历
- **Method**: GET / POST
- **Path**: `/api/basic/calendar`
- **Description**: 获取交易日历数据。支持按日期范围查询。

**请求参数 (Request Parameters)**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `start_time` | string | 是 | 开始日期 (YYYY-MM-DD) |
| `end_time` | string | 是 | 结束日期 (YYYY-MM-DD) |

**响应参数 (Response Parameters)**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `date` | string | 日历日期 |
| `is_open` | integer | 是否交易日 (0:休市, 1:交易) |

**响应示例 (Response Example)**
```json
{
  "code": 200,
  "msg": "Success",
  "data": [
    {
      "date": "2026-01-01",
      "is_open": 0
    },
    {
      "date": "2026-01-02",
      "is_open": 1
    }
  ]
}
```

### 3.2 港股 (HK Stock)

#### 3.2.1 获取港股列表
- **Method**: GET
- **Path**: `/api/stock/hk/list`
- **Description**: 获取所有港股基础信息列表。

**请求参数**: 无

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 股票代码 |
| `name` | string | 股票简称 |
| `fullname` | string | 公司全称 |
| `enname` | string | 英文名称 |
| `cn_spell` | string | 拼音 |
| `market` | string | 市场类别 |
| `list_status` | string | 上市状态 |
| `list_date` | string | 上市日期 |
| `delist_date` | string | 退市日期 |
| `trade_unit` | float | 交易单位 |

**响应示例**
```json
{
  "code": 200,
  "msg": "Success",
  "data": [
    {
      "stock_code": "00700.HK",
      "name": "腾讯控股",
      "fullname": "腾讯控股有限公司",
      "enname": "Tencent Holdings Ltd.",
      "cn_spell": "TXKG",
      "market": "HK",
      "list_status": "L",
      "list_date": "2004-06-16",
      "delist_date": null,
      "trade_unit": 100.0
    }
  ]
}
```

#### 3.2.2 获取互通数据
- **Method**: GET
- **Path**: `/api/stock/hk/connect`
- **Description**: 获取港深上三个市场的互通情况。返回最新的交易日期数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `type` | string | 否 | 互通类型: HK_SZ (深股通), SZ_HK (港股通-深), HK_SH (沪股通), SH_HK (港股通-沪) |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 股票代码 |
| `trade_date` | string | 交易日期 |
| `type` | string | 类型 |
| `name` | string | 股票名称 |
| `type_name` | string | 类型名称 |

**响应示例**
```json
{
  "code": 200,
  "msg": "Success",
  "data": [
    {
      "stock_code": "00700.HK",
      "trade_date": "2023-10-27",
      "type": "SZ_HK",
      "name": "腾讯控股",
      "type_name": "港股通(深>港)"
    }
  ]
}
```

#### 3.2.3 获取港股财务数据
- **Method**: POST
- **Path**: `/api/stock/hk/finance`
- **Description**: 获取港股财务数据，支持按股票代码、时间范围筛选。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 股票代码，支持单个字符串或字符串数组 |
| `start_date` | string | 是 | 开始日期 (YYYY-MM-DD) |
| `end_date` | string | 是 | 结束日期 (YYYY-MM-DD) |
| `page` | integer | 否 | 页码，默认0 |
| `page_size` | integer | 否 | 每页数量，默认10000 |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 股票代码 |
| `name` | string | 股票名称 |
| `end_date` | string | 截止日期 |
| `ind_type` | string | 行业类型 |
| `operate_income` | float | 营业收入 |
| `basic_eps` | float | 基本每股收益 |

**响应示例**
```json
{
  "code": 200,
  "msg": "success",
  "data": [
    {
      "stock_code": "00700",
      "name": "腾讯控股",
      "end_date": "2023-12-31",
      "ind_type": "互联网",
      "report_type": "年报",
      "std_report_date": "2023-12-31",
      "operate_income": 1000000000,
      "basic_eps": 10.5,
      "create_time": "2024-01-01 10:00:00"
    }
  ]
}
```

### 3.3 可转债 (Convertible Bond)

#### 3.3.1 获取分时数据
- **Method**: POST
- **Path**: `/api/bond/history`
- **Description**: 获取可转债的历史分时交易数据。支持1分钟、5分钟原始数据，以及15/30/60分钟聚合数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 可转债代码 |
| `level` | string | 是 | 数据级别: "1min", "5min", "15min", "30min", "60min" |
| `start_time` | string | 是 | 开始时间 (YYYY-MM-DD HH:MM:SS) |
| `end_time` | string | 是 | 结束时间 (YYYY-MM-DD HH:MM:SS) |
| `page` | integer | 否 | 页码 |
| `page_size` | integer | 否 | 每页数量，最大 60000 |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 可转债代码 |
| `trade_time` | string | 交易时间 |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价 |
| `vol` | float | 成交量 |
| `amount` | float | 成交额 |

#### 3.3.2 获取日K线数据
- **Method**: POST
- **Path**: `/api/bond/daily`
- **Description**: 获取可转债的日K线历史数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 可转债代码 |
| `start_time` | string | 是 | 开始日期 (YYYY-MM-DD) |
| `end_time` | string | 是 | 结束日期 (YYYY-MM-DD) |
| `page` | integer | 否 | 页码 |
| `page_size` | integer | 否 | 每页数量，最大 60000 |

#### 3.3.3 获取日指标数据
- **Method**: POST
- **Path**: `/api/bond/indicator_daily`
- **Description**: 获取可转债的日指标数据（溢价率、转股价值等）。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 可转债代码 |
| `start_date` | string | 否 | 开始日期 |
| `end_date` | string | 否 | 结束日期 |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 可转债代码 |
| `trade_date` | string | 交易日期 |
| `remain_size` | float | 剩余规模(亿) |
| `pure_bond` | float | 纯债价值 |
| `pure_premium` | float | 纯债溢价率(%) |
| `conv_value` | float | 转股价值 |
| `conv_premium` | float | 转股溢价率(%) |

#### 3.3.4 收盘快照
- **Method**: POST
- **Path**: `/api/bond/closing_snapshot`
- **Description**: 获取可转债收盘时的快照数据。

#### 3.3.5 获取可转债列表
- **Method**: POST
- **Path**: `/api/bond/list`
- **Description**: 获取可转债基本信息列表，包含债券代码、名称、转股信息、利率条款等详细信息。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `bond_code` | string\|array | 否 | 可转债代码 |
| `stock_code` | string\|array | 否 | 正股代码 |
| `exchange` | string | 否 | 交易所代码 |

### 3.4 ETF

#### 3.4.0 获取ETF实时分时数据
- **Method**: POST
- **Path**: `/api/etf/realtime/history`
- **Description**: 获取全市场或指定ETF实时 1 分钟级别分时数据（支持查询最近7天内的数据）。返回数据会根据 stock_code + trade_time 进行去重。注意：`stock_code` 或 `trade_time` 至少提供一个。

**请求参数 (Request Parameters)**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string \| string[] | 否 | ETF代码，如 159001.SZ 或 ["159001.SZ", "510050.SH"] (必须提供 stock_code 或 trade_time 之一) |
| `trade_time` | string | 否 | 交易时间，如 2026-03-15 09:31:00 (必须提供 stock_code 或 trade_time 之一) |
| `date` | string | 否 | 日期，格式 YYYY-MM-DD，默认今天，支持查询最近7天内的数据 |

**响应参数 (Response Parameters)**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | ETF代码 |
| `trade_time` | string | 交易时间 |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价 |
| `vol` | float | 成交量 |
| `amount` | float | 成交额 |

**响应示例 (Response Example)**
```json
{
  "code": 200,
  "msg": "成功",
  "data": {
    "date": "2026-03-15",
    "count": 1,
    "list": [
      {
        "stock_code": "159001.SZ",
        "trade_time": "2026-03-15 09:31:00",
        "open": 1.05,
        "high": 1.06,
        "low": 1.04,
        "close": 1.055,
        "vol": 10000,
        "amount": 10500
      }
    ]
  }
}
```

#### 3.4.1 获取分时数据
- **Method**: POST
- **Path**: `/api/etf/history`
- **Description**: 获取ETF的历史分时交易数据。

#### 3.4.2 获取日K线数据
- **Method**: POST
- **Path**: `/api/etf/daily`
- **Description**: 获取ETF的日K线历史数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | ETF代码 |
| `start_time` | string | 是 | 开始日期 (YYYY-MM-DD) |
| `end_time` | string | 是 | 结束日期 (YYYY-MM-DD) |
| `page` | integer | 否 | 页码 |
| `page_size` | integer | 否 | 每页数量，最大 60000 |

**响应字段**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | ETF代码 |
| `trade_date` | string | 交易日期 |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价 |
| `pre_close` | float | 昨收价 |
| `change` | float | 涨跌额 |
| `pct_chg` | float | 涨跌幅(%) |
| `vol` | float | 成交量 |
| `amount` | float | 成交额(千元) |
| `prev_close` | float | 前收盘价 |
| `num_trades` | float | 成交笔数 |
| `iopv` | float | 基金份额参考净值(IOPV) |
| `total_shares` | float | 总份额(亿份) |
| `total_assets` | float | 总规模(亿元) |
| `unit_nav` | float | 基金单位净值 |
| `accum_nav` | float | 累计净值 |
| `adj_nav` | float | 复权净值 |

#### 3.4.4 获取日K线数据(前复权)
- **Method**: POST
- **Path**: `/api/etf/daily_adj`
- **Description**: 获取指定ETF的前复权日K线数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string | 否 | ETF代码 |
| `start_time` | string | 是 | 开始时间 |
| `end_time` | string | 是 | 结束时间 |
| `page` | integer | 否 | 页码 |
| `page_size` | integer | 否 | 每页数量，最大 60000 |

#### 3.4.5 获取分钟K线数据(复权)
- **Method**: POST
- **Path**: `/api/etf/min_adj`
- **Description**: 获取指定ETF的复权分钟K线数据。目前仅支持前复权(qfq)。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string | 是 | ETF代码 |
| `level` | string | 是 | 数据级别: "1min", "5min", "15min", "30min", "60min" |
| `start_time` | string | 是 | 开始时间 |
| `end_time` | string | 是 | 结束时间 |
| `page` | integer | 否 | 页码 |
| `page_size` | integer | 否 | 每页数量，最大 60000 |

#### 3.4.6 获取复权因子
- **Method**: POST
- **Path**: `/api/etf/adj_factor`
- **Description**: 获取指定ETF的复权因子数据。

#### 3.4.7 复权因子变更查询
- **Method**: POST
- **Path**: `/api/etf/adj_factor/changes`
- **Description**: 查询指定日期是否有复权因子变更。

### 3.5 股票 (Stock)

#### 3.5.1 获取每日财务数据
- **Method**: POST
- **Path**: `/api/stock/finance`
- **Description**: 获取指定股票的每日财务指标数据（市盈率、市净率、换手率等）。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 股票代码 |
| `start_time` | string | 否 | 开始日期 |
| `end_time` | string | 否 | 结束日期 |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `pe` | float | 市盈率 |
| `pe_ttm` | float | 市盈率TTM |
| `pb` | float | 市净率 |
| `total_mv` | float | 总市值 |

#### 3.5.1.1 获取财务指标报表数据
- **Method**: POST
- **Path**: `/api/stock/financial_indicator`
- **Description**: 从 `stock_financial_indicator` 表获取财务指标数据。`stock_code`、`end_date`、`ann_date` 三个参数至少填写一个。单次请求最多返回 10000 条数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 股票代码，例如 `600000.SH`，支持数组 |
| `end_date` | string | 否 | 报告期最后日期，格式 `YYYY-MM-DD` |
| `ann_date` | string | 否 | 公告日期，格式 `YYYY-MM-DD` |
| `page` | integer | 否 | 页码，从0开始 (默认0) |
| `page_size` | integer | 否 | 每页数量 (默认10000，最大10000) |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 股票代码 |
| `ann_date` | string | 公告日期 |
| `end_date` | string | 报告期最后日期 |
| `eps` | float | 每股收益 |
| `dt_eps` | float | 扣非每股收益 |
| `revenue_ps` | float | 每股营业收入 |
| `profit_dedt` | float | 扣除非经常性损益后的净利润 |
| `op_income` | float | 营业利润 |
| `ebit` | float | 息税前利润 |
| `ebitda` | float | 息税折旧摊销前利润 |
| `roe` | float | 净资产收益率 |
| `roe_dt` | float | 扣非净资产收益率 |
| `roe_yearly` | float | 年化净资产收益率 |
| `roa` | float | 总资产净利率 |
| `roa_yearly` | float | 年化总资产净利率 |
| `bps` | float | 每股净资产 |
| `ocfps` | float | 每股经营活动现金流 |
| `cfps` | float | 每股现金流量净额 |
| `grossprofit_margin` | float | 销售毛利率 |
| `gross_margin` | float | 毛利率 |
| `netprofit_margin` | float | 销售净利率 |
| `current_ratio` | float | 流动比率 |
| `quick_ratio` | float | 速动比率 |
| `debt_to_assets` | float | 资产负债率 |
| `basic_eps_yoy` | float | 基本每股收益同比增长率 |
| `netprofit_yoy` | float | 净利润同比增长率 |
| `dt_netprofit_yoy` | float | 扣非净利润同比增长率 |
| `tr_yoy` | float | 营业总收入同比增长率 |
| `or_yoy` | float | 营业收入同比增长率 |
| `q_sales_yoy` | float | 单季度营收同比增长率 |
| `q_netprofit_yoy` | float | 单季度净利润同比增长率 |
| `rd_exp` | float | 研发费用 |

说明：该接口返回 `stock_financial_indicator` 全字段（除 `update_flag`、`create_time`），上表为常用重点字段。

**完整返回字段**
`stock_code, ann_date, end_date, eps, dt_eps, total_revenue_ps, revenue_ps, capital_rese_ps, surplus_rese_ps, undist_profit_ps, extra_item, profit_dedt, gross_margin, current_ratio, quick_ratio, cash_ratio, invturn_days, arturn_days, inv_turn, ar_turn, ca_turn, fa_turn, assets_turn, op_income, valuechange_income, interst_income, daa, ebit, ebitda, fcff, fcfe, current_exint, noncurrent_exint, interestdebt, netdebt, tangible_asset, working_capital, networking_capital, invest_capital, retained_earnings, diluted2_eps, bps, ocfps, retainedps, cfps, ebit_ps, fcff_ps, fcfe_ps, netprofit_margin, grossprofit_margin, cogs_of_sales, expense_of_sales, profit_to_gr, saleexp_to_gr, adminexp_of_gr, finaexp_of_gr, impai_ttm, gc_of_gr, op_of_gr, ebit_of_gr, roe, roe_waa, roe_dt, roa, npta, roic, roe_yearly, roa2_yearly, roe_avg, opincome_of_ebt, investincome_of_ebt, n_op_profit_of_ebt, tax_to_ebt, dtprofit_to_profit, salescash_to_or, ocf_to_or, ocf_to_opincome, capitalized_to_da, debt_to_assets, assets_to_eqt, dp_assets_to_eqt, ca_to_assets, nca_to_assets, tbassets_to_totalassets, int_to_talcap, eqt_to_talcapital, currentdebt_to_debt, longdeb_to_debt, ocf_to_shortdebt, debt_to_eqt, eqt_to_debt, eqt_to_interestdebt, tangibleasset_to_debt, tangasset_to_intdebt, tangibleasset_to_netdebt, ocf_to_debt, ocf_to_interestdebt, ocf_to_netdebt, ebit_to_interest, longdebt_to_workingcapital, ebitda_to_debt, turn_days, roa_yearly, roa_dp, fixed_assets, profit_prefin_exp, non_op_profit, op_to_ebt, nop_to_ebt, ocf_to_profit, cash_to_liqdebt, cash_to_liqdebt_withinterest, op_to_liqdebt, op_to_debt, roic_yearly, total_fa_trun, profit_to_op, q_opincome, q_investincome, q_dtprofit, q_eps, q_netprofit_margin, q_gsprofit_margin, q_exp_to_sales, q_profit_to_gr, q_saleexp_to_gr, q_adminexp_to_gr, q_finaexp_to_gr, q_impair_to_gr_ttm, q_gc_to_gr, q_op_to_gr, q_roe, q_dt_roe, q_npta, q_opincome_to_ebt, q_investincome_to_ebt, q_dtprofit_to_profit, q_salescash_to_or, q_ocf_to_sales, q_ocf_to_or, basic_eps_yoy, dt_eps_yoy, cfps_yoy, op_yoy, ebt_yoy, netprofit_yoy, dt_netprofit_yoy, ocf_yoy, roe_yoy, bps_yoy, assets_yoy, eqt_yoy, tr_yoy, or_yoy, q_gr_yoy, q_gr_qoq, q_sales_yoy, q_sales_qoq, q_op_yoy, q_op_qoq, q_profit_yoy, q_profit_qoq, q_netprofit_yoy, q_netprofit_qoq, equity_yoy, rd_exp`

**响应示例**
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 1,
    "list": [
      {
        "stock_code": "600000.SH",
        "ann_date": "2026-03-28",
        "end_date": "2025-12-31",
        "eps": 1.24,
        "dt_eps": 1.18,
        "revenue_ps": 8.75,
        "profit_dedt": 21000000000,
        "op_income": 32500000000,
        "ebit": 35600000000,
        "ebitda": 40900000000,
        "roe": 11.5,
        "roe_dt": 10.8,
        "roe_yearly": 11.2,
        "roa": 0.82,
        "roa_yearly": 0.89,
        "bps": 12.88,
        "ocfps": 2.36,
        "cfps": 2.11,
        "grossprofit_margin": 33.2,
        "gross_margin": 32.65,
        "netprofit_margin": 12.4,
        "current_ratio": 1.21,
        "quick_ratio": 1.08,
        "debt_to_assets": 74.3,
        "basic_eps_yoy": 9.8,
        "netprofit_yoy": 12.6,
        "dt_netprofit_yoy": 11.4,
        "tr_yoy": 7.2,
        "or_yoy": 6.9,
        "q_sales_yoy": 8.1,
        "q_netprofit_yoy": 10.7,
        "rd_exp": 1850000000
      }
    ]
  }
}
```

#### 3.5.1.2 获取利润表数据
- **Method**: POST
- **Path**: `/api/stock/income`
- **Description**: 从 `stock_income` 表获取利润表数据。`stock_code`、`end_date`、`ann_date` 三个参数至少填写一个。单次请求最多返回 10000 条数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 股票代码，例如 `600000.SH`，支持数组 |
| `end_date` | string | 否 | 报告期最后日期，格式 `YYYY-MM-DD` |
| `ann_date` | string | 否 | 公告日期，格式 `YYYY-MM-DD` |
| `page` | integer | 否 | 页码，从0开始 (默认0) |
| `page_size` | integer | 否 | 每页数量 (默认10000，最大10000) |

**响应参数（常用）**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 股票代码 |
| `ann_date` | string | 公告日期 |
| `end_date` | string | 报告期最后日期 |
| `total_revenue` | float | 营业总收入 |
| `revenue` | float | 营业收入 |
| `operate_profit` | float | 营业利润 |
| `total_profit` | float | 利润总额 |
| `n_income` | float | 净利润(含少数股东损益) |
| `n_income_attr_p` | float | 归母净利润 |
| `basic_eps` | float | 基本每股收益 |
| `diluted_eps` | float | 稀释每股收益 |

说明：该接口返回 `stock_income` 全字段（除 `update_flag`、`create_time`），上表为常用重点字段。

**完整返回字段**
`stock_code, ann_date, f_ann_date, end_date, report_type, comp_type, end_type, basic_eps, diluted_eps, total_revenue, revenue, int_income, prem_earned, comm_income, n_commis_income, n_oth_income, n_oth_b_income, prem_income, out_prem, une_prem_reser, reins_income, n_sec_tb_income, n_sec_uw_income, n_asset_mg_income, oth_b_income, fv_value_chg_gain, invest_income, ass_invest_income, forex_gain, total_cogs, oper_cost, int_exp, comm_exp, biz_tax_surchg, sell_exp, admin_exp, fin_exp, assets_impair_loss, prem_refund, compens_payout, reser_insur_liab, div_payt, reins_exp, oper_exp, compens_payout_refu, insur_reser_refu, reins_cost_refund, other_bus_cost, operate_profit, non_oper_income, non_oper_exp, nca_disploss, total_profit, income_tax, n_income, n_income_attr_p, minority_gain, oth_compr_income, t_compr_income, compr_inc_attr_p, compr_inc_attr_m_s, ebit, ebitda, insurance_exp, undist_profit, distable_profit, rd_exp, fin_exp_int_exp, fin_exp_int_inc, transfer_surplus_rese, transfer_housing_imprest, transfer_oth, adj_lossgain, withdra_legal_surplus, withdra_legal_pubfund, withdra_biz_devfund, withdra_rese_fund, withdra_oth_ersu, workers_welfare, distr_profit_shrhder, prfshare_payable_dvd, comshare_payable_dvd, capit_comstock_div, continued_net_profit`

#### 3.5.1.3 获取资产负债表数据
- **Method**: POST
- **Path**: `/api/stock/balancesheet`
- **Description**: 从 `stock_balancesheet` 表获取资产负债表数据。`stock_code`、`end_date`、`ann_date` 三个参数至少填写一个。单次请求最多返回 10000 条数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 股票代码，例如 `600000.SH`，支持数组 |
| `end_date` | string | 否 | 报告期最后日期，格式 `YYYY-MM-DD` |
| `ann_date` | string | 否 | 公告日期，格式 `YYYY-MM-DD` |
| `page` | integer | 否 | 页码，从0开始 (默认0) |
| `page_size` | integer | 否 | 每页数量 (默认10000，最大10000) |

**响应参数（常用）**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 股票代码 |
| `ann_date` | string | 公告日期 |
| `end_date` | string | 报告期最后日期 |
| `total_assets` | float | 资产总计 |
| `total_cur_assets` | float | 流动资产合计 |
| `total_nca` | float | 非流动资产合计 |
| `total_liab` | float | 负债合计 |
| `total_cur_liab` | float | 流动负债合计 |
| `total_hldr_eqy_exc_min_int` | float | 归属于母公司股东权益合计 |

说明：该接口返回 `stock_balancesheet` 全字段（除 `update_flag`、`create_time`），上表为常用重点字段。

**完整返回字段**
`stock_code, ann_date, f_ann_date, end_date, report_type, comp_type, end_type, total_share, cap_rese, undistr_porfit, surplus_rese, special_rese, money_cap, trad_asset, notes_receiv, accounts_receiv, oth_receiv, prepayment, div_receiv, int_receiv, inventories, amor_exp, nca_within_1y, sett_rsrv, loanto_oth_bank_fi, premium_receiv, reinsur_receiv, reinsur_res_receiv, pur_resale_fa, oth_cur_assets, total_cur_assets, fa_avail_for_sale, htm_invest, lt_eqt_invest, invest_real_estate, time_deposits, oth_assets, lt_rec, fix_assets, cip, const_materials, fixed_assets_disp, produc_bio_assets, oil_and_gas_assets, intan_assets, r_and_d, goodwill, lt_amor_exp, defer_tax_assets, decr_in_disbur, oth_nca, total_nca, cash_reser_cb, depos_in_oth_bfi, prec_metals, deriv_assets, rr_reins_une_prem, rr_reins_outstd_cla, rr_reins_lins_liab, rr_reins_lthins_liab, refund_depos, ph_pledge_loans, refund_cap_depos, indep_acct_assets, client_depos, client_prov, transac_seat_fee, invest_as_receiv, total_assets, lt_borr, st_borr, cb_borr, depos_ib_deposits, loan_oth_bank, trading_fl, notes_payable, acct_payable, adv_receipts, sold_for_repur_fa, comm_payable, payroll_payable, taxes_payable, int_payable, div_payable, oth_payable, acc_exp, deferred_inc, st_bonds_payable, payable_to_reinsurer, rsrv_insur_cont, acting_trading_sec, acting_uw_sec, non_cur_liab_due_1y, oth_cur_liab, total_cur_liab, bond_payable, lt_payable, specific_payables, estimated_liab, defer_tax_liab, defer_inc_non_cur_liab, oth_ncl, total_ncl, depos_oth_bfi, deriv_liab, depos, agency_bus_liab, oth_liab, prem_receiv_adva, depos_received, ph_invest, reser_une_prem, reser_outstd_claims, reser_lins_liab, reser_lthins_liab, indept_acc_liab, pledge_borr, indem_payable, policy_div_payable, total_liab, treasury_share, ordin_risk_reser, forex_dinc_min_int, total_liab_hldr_eqy, lt_payroll_payable, oth_comp_income, oth_eqt_tools, oth_eqt_tools_p_shr, lending_funds, acc_receivable, st_fin_payable, payables, hfs_assets, hfs_sales, cost_fin_assets, fair_value_fin_assets, contract_assets, contract_liab, accounts_receiv_bill, accounts_pay, oth_rcv_total, fix_assets_total, cip_total, oth_pay_total, long_pay_total, debt_invest, oth_debt_invest, total_hldr_eqy_exc_min_int`

#### 3.5.1.4 获取现金流量表数据
- **Method**: POST
- **Path**: `/api/stock/cashflow`
- **Description**: 从 `stock_cashflow` 表获取现金流量表数据。`stock_code`、`end_date`、`ann_date` 三个参数至少填写一个。单次请求最多返回 10000 条数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 股票代码，例如 `600000.SH`，支持数组 |
| `end_date` | string | 否 | 报告期最后日期，格式 `YYYY-MM-DD` |
| `ann_date` | string | 否 | 公告日期，格式 `YYYY-MM-DD` |
| `page` | integer | 否 | 页码，从0开始 (默认0) |
| `page_size` | integer | 否 | 每页数量 (默认10000，最大10000) |

**响应参数（常用）**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 股票代码 |
| `ann_date` | string | 公告日期 |
| `end_date` | string | 报告期最后日期 |
| `n_cashflow_act` | float | 经营活动产生的现金流量净额 |
| `n_cashflow_inv_act` | float | 投资活动产生的现金流量净额 |
| `n_cash_flows_fnc_act` | float | 筹资活动产生的现金流量净额 |
| `n_incr_cash_cash_equ` | float | 现金及现金等价物净增加额 |
| `c_cash_equ_end_period` | float | 期末现金及现金等价物余额 |

说明：该接口返回 `stock_cashflow` 全字段（除 `update_flag`、`create_time`），上表为常用重点字段。

**完整返回字段**
`stock_code, ann_date, f_ann_date, end_date, comp_type, report_type, end_type, net_profit, finan_exp, c_fr_sale_sg, recp_tax_rends, n_depos_incr_fi, n_incr_loans_cb, n_inc_borr_oth_fi, prem_fr_orig_contr, n_incr_insured_dep, n_reinsur_prem, n_incr_disp_tfa, ifc_cash_incr, n_incr_disp_faas, n_incr_loans_oth_bank, n_cap_incr_repur, c_fr_oth_operate_a, c_inf_fr_operate_a, c_paid_goods_s, c_paid_to_for_empl, c_paid_for_taxes, n_incr_clt_loan_adv, n_incr_dep_cbob, c_pay_claims_orig_inco, pay_handling_chrg, pay_comm_insur_plcy, oth_cash_pay_oper_act, st_cash_out_act, n_cashflow_act, oth_recp_ral_inv_act, c_disp_withdrwl_invest, c_recp_return_invest, n_recp_disp_fiolta, n_recp_disp_sobu, stot_inflows_inv_act, c_pay_acq_const_fiolta, c_paid_invest, n_disp_subs_oth_biz, oth_pay_ral_inv_act, n_incr_pledge_loan, stot_out_inv_act, n_cashflow_inv_act, c_recp_borrow, proc_issue_bonds, oth_cash_recp_ral_fnc_act, stot_cash_in_fnc_act, free_cashflow, c_prepay_amt_borr, c_pay_dist_dpcp_int_exp, incl_dvd_profit_paid_sc_ms, oth_cashpay_ral_fnc_act, stot_cashout_fnc_act, n_cash_flows_fnc_act, eff_fx_flu_cash, n_incr_cash_cash_equ, c_cash_equ_beg_period, c_cash_equ_end_period, c_recp_cap_contrib, incl_cash_rec_saims, uncon_invest_loss, prov_depr_assets, depr_fa_coga_dpba, amort_intang_assets, lt_amort_deferred_exp, decr_deferred_exp, incr_acc_exp, loss_disp_fiolta, loss_scr_fa, loss_fv_chg, invest_loss, decr_def_inc_tax_assets, incr_def_inc_tax_liab, decr_inventories, decr_oper_payable, incr_oper_payable, others, im_net_cashflow_oper_act, conv_debt_into_cap, conv_copbonds_due_within_1y, fa_fnc_leases, im_n_incr_cash_equ, net_dism_capital_add, net_cash_rece_sec, credit_impa_loss, use_right_asset_dep, oth_loss_asset, end_bal_cash, beg_bal_cash, end_bal_cash_equ, beg_bal_cash_equ`

#### 3.5.2.1 获取大小单资金金流向
- **Method**: POST
- **Path**: `/api/stock/main_fund_flow`
- **Description**: 获取大小单资金金流向数据。分档口径：小单 `<5万`，中单 `5万~20万`，大单 `20万~100万`，特大单 `>=100万`。查询条件支持仅传 `stock_code`、仅传 `start_time+end_time` 或两者同时传。时间区间为闭区间，`start_time = end_time` 时可查询当天数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `start_time` | string | 否 | 开始日期 (YYYY-MM-DD)，与 end_time 配套 |
| `end_time` | string | 否 | 结束日期 (YYYY-MM-DD)，与 start_time 配套；闭区间，start_time=end_time 可查当天 |
| `stock_code` | string\|array | 否 | 股票代码，例如 `600000.SH` |
| `page` | integer | 否 | 页码，从0开始 (默认0) |
| `page_size` | integer | 否 | 每页数量 (默认10000，最大10000) |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `trade_date` | string | 交易日期 |
| `stock_code` | string | 股票代码 |
| `buy_sm_vol` | float | 小单买入量（手） |
| `buy_sm_amount` | float | 小单买入金额（万元） |
| `sell_sm_vol` | float | 小单卖出量（手） |
| `sell_sm_amount` | float | 小单卖出金额（万元） |
| `buy_md_vol` | float | 中单买入量（手） |
| `buy_md_amount` | float | 中单买入金额（万元） |
| `sell_md_vol` | float | 中单卖出量（手） |
| `sell_md_amount` | float | 中单卖出金额（万元） |
| `buy_lg_vol` | float | 大单买入量（手） |
| `buy_lg_amount` | float | 大单买入金额（万元） |
| `sell_lg_vol` | float | 大单卖出量（手） |
| `sell_lg_amount` | float | 大单卖出金额（万元） |
| `buy_elg_vol` | float | 特大单买入量（手） |
| `buy_elg_amount` | float | 特大单买入金额（万元） |
| `sell_elg_vol` | float | 特大单卖出量（手） |
| `sell_elg_amount` | float | 特大单卖出金额（万元） |
| `net_mf_vol` | float | 净流入量（手） |
| `net_mf_amount` | float | 净流入额（万元） |

#### 3.5.2.2 获取主力资金流向总览
- **Method**: POST
- **Path**: `/api/stock/main_fund_flow_overview`
- **Description**: 获取主力资金流向总览数据。分档口径：小单 `<5万`，中单 `5万~20万`，大单 `20万~100万`，特大单 `>=100万`。查询条件支持仅传 `stock_code`、仅传 `start_time+end_time` 或两者同时传。时间区间为闭区间，`start_time = end_time` 时可查询当天数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `start_time` | string | 否 | 开始日期 (YYYY-MM-DD)，与 end_time 配套 |
| `end_time` | string | 否 | 结束日期 (YYYY-MM-DD)，与 start_time 配套；闭区间，start_time=end_time 可查当天 |
| `stock_code` | string\|array | 否 | 股票代码，例如 `600000.SH` |
| `page` | integer | 否 | 页码，从0开始 (默认0) |
| `page_size` | integer | 否 | 每页数量 (默认10000，最大10000) |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `trade_date` | string | 交易日期 |
| `stock_code` | string | 股票代码 |
| `name` | string | 股票名称 |
| `close` | float | 收盘价 |
| `pct_change` | float | 涨跌幅 |
| `net_amount` | float | 主力净流入额 |
| `net_amount_rate` | float | 主力净流入率 |
| `buy_elg_amount` | float | 超大单净流入额 |
| `buy_elg_amount_rate` | float | 超大单净流入率 |
| `buy_lg_amount` | float | 大单净流入额 |
| `buy_lg_amount_rate` | float | 大单净流入率 |
| `buy_md_amount` | float | 中单净流入额 |
| `buy_md_amount_rate` | float | 中单净流入率 |
| `buy_sm_amount` | float | 小单净流入额 |
| `buy_sm_amount_rate` | float | 小单净流入率 |

#### 3.5.2.3 获取筹码峰分布
- **Method**: POST
- **Path**: `/api/stock/cyq_chips`
- **Description**: 获取筹码峰分布数据。查询条件支持仅传 `stock_code`、仅传 `start_time+end_time` 或两者同时传。时间区间为闭区间，`start_time = end_time` 时可查询当天数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `start_time` | string | 否 | 开始日期 (YYYY-MM-DD)，与 end_time 配套 |
| `end_time` | string | 否 | 结束日期 (YYYY-MM-DD)，与 start_time 配套；闭区间，start_time=end_time 可查当天 |
| `stock_code` | string\|array | 否 | 股票代码，例如 `600000.SH` |
| `page` | integer | 否 | 页码，从0开始 (默认0) |
| `page_size` | integer | 否 | 每页数量 (默认10000，最大10000) |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `trade_date` | string | 交易日期 |
| `stock_code` | string | 股票代码 |
| `price` | float | 成本价格 |
| `percent` | float | 价格占比 |

#### 获取涨跌停数据
- **Method**: POST
- **Path**: `/api/stock/limit_list`
- **Description**: 获取指定股票（支持多只）的涨跌停数据（封板时间、封单金额、连板状态等）。支持按日期范围筛选。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 股票代码，例如 '600000.SH'。最大支持100个。 |
| `start_time` | string | 是 | 开始日期 (YYYY-MM-DD) |
| `end_time` | string | 是 | 结束日期 (YYYY-MM-DD) |
| `page` | integer | 否 | 页码，从0开始 (默认0) |
| `page_size` | integer | 否 | 每页数量 (默认10000) |

**响应示例**
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 1,
    "list": [
      {
        "trade_date": "2026-01-05",
        "stock_code": "000426.SZ",
        "name": "兴业银锡",
        "industry": "有色金属",
        "close": 39.16,
        "pct_chg": 10.0,
        "amount": 4089780784.0,
        "limit_amount": null,
        "float_mv": 69517045970.24,
        "total_mv": 69533895735.04,
        "turnover_ratio": 6.07,
        "fd_amount": 31512052.0,
        "first_time": "142900",
        "last_time": "145503",
        "open_times": 1,
        "up_stat": "1/1",
        "limit_times": 1.0,
        "limit": "U"
      }
    ]
  }
}
```

#### 获取涨停数据
- **Method**: POST
- **Path**: `/api/stock/limit_up`
- **Description**: 获取指定股票（支持多只）的涨停明细数据（封单、连板、原因等）。支持按日期范围筛选。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 股票代码，例如 '600000.SH' |
| `start_time` | string | 是 | 开始日期 (YYYY-MM-DD) |
| `end_time` | string | 是 | 结束日期 (YYYY-MM-DD) |
| `page` | integer | 否 | 页码，从0开始 (默认0) |
| `page_size` | integer | 否 | 每页数量 (默认10000，最大10000) |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `trade_date` | string | 交易日期 |
| `stock_code` | string | 股票代码 |
| `stock_name` | string | 股票名称 |
| `price` | float | 最新价/收盘价 |
| `change_percent` | float | 涨跌幅(%) |
| `first_limit_time` | string | 首次涨停时间 |
| `final_limit_time` | string | 最终涨停时间 |
| `consecutive_days` | integer | 连续涨停天数 |
| `sealed_volume` | float | 封单量 |
| `sealed_amount` | float | 封单额 |
| `sealed_turnover_ratio` | float | 封成比 |
| `sealed_flow_ratio` | float | 封流比 |
| `open_count` | integer | 开板次数 |
| `boards` | integer | 几天几板中的板数 |
| `limit_type` | string | 涨停类型 |
| `is_limit_up` | integer | 是否涨停(1是0否) |
| `reason_text` | string | 涨停原因 |

**响应示例**
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 1,
    "list": [
      {
        "trade_date": "2026-04-10",
        "stock_code": "603716.SH",
        "stock_name": "塞力医疗",
        "price": 14.59,
        "change_percent": 10.03,
        "first_limit_time": "09:33:11",
        "final_limit_time": "14:56:07",
        "consecutive_days": 2,
        "sealed_volume": 151492,
        "sealed_amount": 22152780.0,
        "sealed_turnover_ratio": 1.89,
        "sealed_flow_ratio": 0.83,
        "open_count": 2,
        "boards": 2,
        "limit_type": "普通涨停",
        "is_limit_up": 1,
        "reason_text": "医药商业"
      }
    ]
  }
}
```

#### 获取同花顺热度榜 (股票-独立数据)
- **Method**: GET / POST
- **Path**: `/api/ths/hot`
- **Description**: 获取同花顺热度榜数据。支持传 `market` 指定榜单类型，支持传 `trade_date` 指定交易日；不传 `trade_date` 默认返回该市场最新交易日数据。

**请求参数**
| 参数名 | 类型 | 必选 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `market` | string | 否 | `热股` | 热榜类型。可选值：`热股`, `ETF`, `可转债`, `行业板块`, `概念板块`, `期货` |
| `trade_date` | string | 否 | - | 指定交易日期，支持 `YYYY-MM-DD` 或 `YYYYMMDD`。GET 用 Query，POST 用 JSON Body |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `trade_date` | string | 交易日期 |
| `update_time` | string | 排行榜更新时间 |
| `list` | array | 热榜数据列表 |
| `└─ name` | string | 名称 |
| `└─ code` | string | 代码 |
| `└─ rank` | integer | 排名 |
| `└─ pct_change` | float | 涨跌幅(%) |
| `└─ hot` | float | 热度值 |

#### 获取券商盈利预测 (股票-独立数据)
- **Method**: POST
- **Path**: `/api/stock/report_rc`
- **Description**: 查询券商盈利预测数据。`stock_code`、`report_date`、`start_date+end_date` 三种查询条件至少传一个；若传时间区间，`start_date` 与 `end_date` 必须同时传入。参数使用 JSON Body 传递。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string | 否 | 股票代码，例如 `600000.SH` |
| `report_date` | string | 否 | 报告日期，支持 `YYYY-MM-DD` 或 `YYYYMMDD` |
| `start_date` | string | 否 | 开始日期，支持 `YYYY-MM-DD` 或 `YYYYMMDD`（与 `end_date` 配套） |
| `end_date` | string | 否 | 结束日期，支持 `YYYY-MM-DD` 或 `YYYYMMDD`（与 `start_date` 配套） |
| `page` | integer | 否 | 页码，从0开始，默认0 |
| `page_size` | integer | 否 | 每页数量，默认10000，最大10000 |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 股票代码 |
| `name` | string | 股票名称 |
| `report_date` | string | 研报日期 |
| `report_title` | string | 报告标题 |
| `report_type` | string | 报告类型 |
| `classify` | string | 报告分类 |
| `org_name` | string | 机构名称 |
| `author_name` | string | 作者 |
| `quarter` | string | 预测报告期 |
| `op_rt` | float | 预测营业收入（万元） |
| `op_pr` | float | 预测营业利润（万元） |
| `tp` | float | 预测利润总额（万元） |
| `np` | float | 预测净利润（万元） |
| `eps` | float | 预测每股收益（元） |
| `pe` | float | 预测市盈率 |
| `rd` | float | 预测股息率 |
| `roe` | float | 预测净资产收益率 |
| `ev_ebitda` | float | 预测EV/EBITDA |
| `rating` | string | 卖方评级 |
| `max_price` | float | 预测最高目标价 |
| `min_price` | float | 预测最低目标价 |
| `imp_dg` | string | 机构关注度 |

#### 获取ST信息
- **Method**: POST
- **Path**: `/api/stock/st_info`
- **Description**: 获取指定股票的ST信息或按日期获取ST信息列表。股票代码或时间范围必须至少填一个。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 股票代码，例如 '600069.SH' |
| `start_time` | string | 否 | 开始日期 (YYYY-MM-DD) |
| `end_time` | string | 否 | 结束日期 (YYYY-MM-DD) |
| `page` | integer | 否 | 页码，从0开始 (默认0) |
| `page_size` | integer | 否 | 每页数量 (默认10000) |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 股票代码 |
| `trade_date` | string | 交易日期 |
| `stock_name` | string | 股票名称 |
| `type` | string | 类型 (例如 ST) |
| `type_name` | string | 类型名称 (例如 风险警示板) |

**响应示例**
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 1,
    "list": [
      {
        "stock_code": "600069.SH",
        "trade_date": "2020-04-30",
        "stock_name": "*ST银鸽",
        "type": "ST",
        "type_name": "风险警示板"
      }
    ]
  }
}
```

#### 3.5.2 获取股票列表
- **Method**: GET
- **Path**: `/api/stock/list`
- **Description**: 获取所有股票的基础信息列表。

#### 3.5.3 获取日K线数据
- **Method**: POST
- **Path**: `/api/stock/daily`
- **Description**: 获取指定股票的日线级别行情数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 股票代码，例如 "600000.SH" |
| `start_time` | string | 是 | 开始时间 (YYYY-MM-DD) |
| `end_time` | string | 是 | 结束时间 (YYYY-MM-DD) |
| `volType` | string | 否 | 成交量单位，可选 `share`(股，默认) 或 `lot`(手) |
| `page` | integer | 否 | 页码 |
| `page_size` | integer | 否 | 每页数量 |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `trade_date` | string | 交易日期 |
| `stock_code` | string | 股票代码 |
| `stock_name` | string | 股票名称 |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价 |
| `pre_close` | float | 昨收价 |
| `change` | float | 涨跌额 |
| `pct_chg` | float | 涨跌幅(%) |
| `vol` | float | 成交量（单位由 `volType` 决定：`share`=股，`lot`=手） |
| `amount` | float | 成交额 |

#### 获取港股日线数据
- **Method**: POST
- **Path**: `/api/stock/hk/daily`
- **Description**: 获取指定港股的日线级别行情数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 股票代码，例如 "00700.HK" |
| `start_time` | string | 是 | 开始时间 (YYYY-MM-DD) |
| `end_time` | string | 是 | 结束时间 (YYYY-MM-DD) |
| `volType` | string | 否 | 成交量单位，可选 `share`(股，默认) 或 `lot`(手) |
| `page` | integer | 否 | 页码 |
| `page_size` | integer | 否 | 每页数量 |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `trade_date` | string | 交易日期 |
| `stock_code` | string | 股票代码 |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价 |
| `pre_close` | float | 前收盘价 |
| `change` | float | 涨跌额 |
| `pct_chg` | float | 涨跌幅(%) |
| `pe` | float | 市盈率 |
| `pe_percentile` | float | 市盈率百分位(%) |
| `vol` | float | 成交量（单位由 `volType` 决定：`share`=股，`lot`=手） |
| `amount` | float | 成交额 |
| `currency` | string | 交易币种 |

#### 3.5.4 获取日K线数据(复权)
- **Method**: POST
- **Path**: `/api/stock/daily_adj`
- **Description**: 获取指定股票的复权日K线数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string | 否 | 股票代码，例如 "600000.SH" |
| `start_time` | string | 否 | 开始时间 (YYYY-MM-DD) |
| `end_time` | string | 否 | 结束时间 (YYYY-MM-DD) |
| `algo` | string | 否 | 复权算法，recursive为动态复权（默认），factor为静态因子复权 |
| `volType` | string | 否 | 成交量单位，可选 `share`(股，默认) 或 `lot`(手) |
| `page` | integer | 否 | 页码 |
| `page_size` | integer | 否 | 每页数量 |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `trade_date` | string | 交易日期 |
| `stock_code` | string | 股票代码 |
| `stock_name` | string | 股票名称 |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价 |
| `change` | float | 涨跌额 |
| `pct_chg` | float | 涨跌幅(%) |
| `vol` | float | 成交量（单位由 `volType` 决定：`share`=股，`lot`=手） |
| `amount` | float | 成交额 |

#### 3.5.5 获取周期K线数据
- **Method**: POST
- **Path**: `/api/stock/kline`
- **Description**: 获取指定股票的周期K线(周/月)级别行情数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `period` | string | 是 | K线周期，可选值: weekly, monthly |
| `stock_code` | string\|array | 否 | 股票代码，例如 "600000.SH" |
| `start_time` | string | 是 | 开始时间 (YYYY-MM-DD) |
| `end_time` | string | 是 | 结束时间 (YYYY-MM-DD) |
| `page` | integer | 否 | 页码 |
| `page_size` | integer | 否 | 每页数量 |

**响应参数**同日K线数据，其中 `trade_date` 为该周期最后交易日。

#### 3.5.6 获取周期K线数据(复权)
- **Method**: POST
- **Path**: `/api/stock/kline_adj`
- **Description**: 获取指定股票的复权周期K线(周/月)数据。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `period` | string | 是 | K线周期，可选值: weekly, monthly |
| `stock_code` | string\|array | 否 | 股票代码，例如 "600000.SH" |
| `start_time` | string | 否 | 开始时间 (YYYY-MM-DD) |
| `end_time` | string | 否 | 结束时间 (YYYY-MM-DD) |
| `algo` | string | 否 | 复权算法，recursive为动态复权（默认），factor为静态因子复权 |
| `page` | integer | 否 | 页码 |
| `page_size` | integer | 否 | 每页数量 |

**响应参数**同日K线复权数据，其中 `trade_date` 为该周期最后交易日。

#### 3.5.5 获取分钟K线数据(复权)
- **Method**: POST
- **Path**: `/api/stock/min_adj`
- **Description**: 获取指定股票的分钟级复权K线数据。

#### 3.5.6 获取复权因子
- **Method**: POST
- **Path**: `/api/stock/adj_factor`
- **Description**: 获取指定股票的复权因子数据。

#### 3.5.7 复权因子变更查询
- **Method**: POST
- **Path**: `/api/stock/adj_factor/changes`
- **Description**: 查询指定日期是否有复权因子变更。

#### 3.5.8 获取历史分时
- **Method**: POST
- **Path**: `/api/stock/history`
- **Description**: 获取指定股票的历史分时数据。

#### 3.5.9 下载全市场当天全部分时
- **Method**: POST
- **Path**: `/api/stock/daily_dump`
- **Description**: 下载全市场当天的全部1分钟或5分钟分时数据，或日线数据。
- **说明**: 数据量比较多，一个日期一天最多只能下载10次，超过后会被禁止下载该日期三天，请联系客服解封。

#### 3.5.10 获取股票停牌信息
- **Method**: GET
- **Path**: `/api/stock/suspension`
- **Description**: 获取股票停牌信息。

#### 3.5.11 竞价结果快照
- **Method**: POST
- **Path**: `/api/stock/call_auction`
- **Description**: 获取集合竞价时段的数据。

#### 3.5.12 收盘快照
- **Method**: POST
- **Path**: `/api/stock/closing_snapshot`
- **Description**: 获取收盘时的快照数据。

#### 3.5.13 每日快照数据
- **Method**: POST
- **Path**: `/api/stock/snapshot_daily`
- **Description**: 实时更新的股票快照数据（约 10 秒更新一次）。用于获取最新价、开盘价、昨收价等日级别快照字段。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 股票代码 |
| `date` | string | 否 | 日期 (YYYY-MM-DD)，日期和股票代码至少提供一个 |
| `page` | integer | 否 | 页码 |
| `page_size` | integer | 否 | 每页数量 |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 股票代码 |
| `update_time` | string | 更新时间 |
| `last_price` | float | 最新价 |
| `open_price` | float | 开盘价 |
| `high_price` | float | 最高价 |
| `low_price` | float | 最低价 |
| `prev_close_price` | float | 昨收价 |
| `volume` | int64 | 成交量 |
| `turnover` | float | 成交额 |
| `turnover_rate` | float | 换手率 |
| `pe_ratio` | float | 市盈率 |
| `pb_ratio` | float | 市净率 |
| `pe_ttm_ratio` | float | 市盈率TTM |

#### 3.5.14 Websocket推送历史
- **Method**: POST
- **Path**: `/api/stock/snapshot_push_history`
- **Description**: 获取当天每一次 WebSocket 推送的历史记录（注意：这不是分钟级数据，也不是秒级数据）。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 股票代码 |
| `start_time` | string | 是 | 开始时间 (YYYY-MM-DD HH:mm:ss) |
| `end_time` | string | 否 | 结束时间 (YYYY-MM-DD HH:mm:ss) |
| `page` | integer | 否 | 页码 |
| `page_size` | integer | 否 | 每页数量 |

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 股票代码 |
| `update_time` | string | 更新时间 |
| `last_price` | float | 最新价 |
| `open_price` | float | 开盘价 |
| `high_price` | float | 最高价 |
| `low_price` | float | 最低价 |
| `prev_close_price` | float | 昨收价 |
| `volume` | int64 | 成交量 |
| `turnover` | float | 成交额（当日累计） |
| `turnover_rate` | float | 换手率 |

#### 3.5.15 股票条件搜索
- **Method**: POST
- **Path**: `/api/stock/search`
- **Description**: 通过自然语言搜索语句查询符合条件的股票。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `query` | string | 是 | 搜索条件，例如：pe_ttm < 20 |
| `stock_code` | string | 否 | 股票代码 |
| `date` | string | 否 | 日期 |


#### 3.5.16 获取龙虎榜机构明细
- **Method**: POST
- **Path**: `/api/stock/dragon_tiger`
- **Description**: 获取龙虎榜机构明细数据。支持按日期或股票代码查询。

**请求参数**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `date` | string | 否 | 交易日期 (YYYY-MM-DD) |
| `stock_code` | string | 否 | 股票代码 |
| `page` | integer | 否 | 页码 (从0开始，默认0) |
| `page_size` | integer | 否 | 每页数量 (默认20) |

> **注意**: `date` 和 `stock_code` 必须至少提供一个。

**响应参数**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `trade_date` | string | 交易日期 |
| `stock_code` | string | 股票代码 |
| `org_name` | string | 机构名称 |
| `buy_amount` | float | 买入金额 |
| `buy_ratio` | float | 买入比例 |
| `sell_amount` | float | 卖出金额 |
| `sell_ratio` | float | 卖出比例 |
| `net_buy_amount`| float | 净买入金额 |
| `direction` | int | 方向 (1: 买入, -1: 卖出, 0: 混合) |
| `reason` | string | 上榜原因 |


### 3.6 指数 (Index)

#### 3.6.1 历史数据 (History Data)

##### 3.6.1.8 获取日K线数据
- **Method**: POST
- **Path**: `/api/index/daily`
- **Description**: 获取指数每日行情数据。单次最多调取8000行记录，可以通过设置 start_date 和 end_date 进行区间查询。

**请求参数 (Request Parameters)**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string \| string[] | 否 | 指数代码，来源指数基础信息接口 |
| `start_date` | string | 否 | 开始日期 (YYYY-MM-DD) |
| `end_date` | string | 否 | 结束日期 (YYYY-MM-DD) |
| `page` | int | 否 | 页码，从0开始 |
| `page_size` | int | 否 | 每页数量，默认2000，最大8000 |

**响应参数 (Response Parameters)**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `ts_code` | string | TS指数代码 |
| `trade_date` | string | 交易日 |
| `close` | float | 收盘点位 |
| `open` | float | 开盘点位 |
| `high` | float | 最高点位 |
| `low` | float | 最低点位 |
| `pre_close` | float | 昨日收盘点 |
| `change` | float | 涨跌点 |
| `pct_chg` | float | 涨跌幅（%） |
| `vol` | float | 成交量（手） |
| `amount` | float | 成交额（元） |

##### 3.6.1.11 获取指数成分和权重
- **Method**: POST
- **Path**: `/api/index/weight`
- **Description**: 获取指数月度成分和权重数据。`index_code` 必传，`stock_code` 可传用于筛选成分股（底层字段 `con_code`）。不传 `trade_date` 时返回该指数最新月份数据；传入 `trade_date` 时返回指定日期所在月份数据。接口仅按 apiKey 是否具备接口权限进行校验。

**Request Parameters (Body, JSON)**:
| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| `index_code` | string | 是 | 指数代码，例如 `000300.SH` |
| `stock_code` | string \| string[] | 否 | 成分股代码（筛选 `con_code`） |
| `trade_date` | string | 否 | 交易日期，支持 `YYYY-MM` 或 `YYYY-MM-DD`；查询时仅按年和月生效 |
| `page` | integer | 否 | 页码，默认 0 |
| `page_size` | integer | 否 | 每页条数，默认 2000，最大 10000 |

**Response**:
```json
{
  "code": 200,
  "msg": "成功",
  "data": {
    "total": 2,
    "list": [
      {
        "index_code": "000300.SH",
        "stock_code": "600519.SH",
        "trade_date": "2026-03-01",
        "weight": 4.123456
      },
      {
        "index_code": "000300.SH",
        "stock_code": "000858.SZ",
        "trade_date": "2026-03-01",
        "weight": 1.987654
      }
    ]
  }
}
```

##### 3.6.1.1 获取TDX板块列表
- **Method**: GET
- **Path**: `/api/tdx/blocks`
- **Description**: 获取通达信板块列表数据。支持按板块名称筛选。板块类型（block_type）为必填参数。

**Request Parameters (Query)**:
| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| block_name | string | 否 | 板块名称 (例如: 5G概念) |
| block_type | integer | 是 | 板块类型 (0:行业板块, 1:风格板块, 2:概念板块, 3:指数板块) |
| page | integer | 否 | 页码，默认 0 |
| page_size | integer | 否 | 每页条数，默认 10000 |

**Response**:
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 100,
    "page": 0,
    "page_size": 10000,
    "list": [
      {
        "block_code": "880506.TDX",
        "block_name": "5G概念",
        "block_type": 2
      }
    ]
  }
}
```

##### 3.6.1.2 获取TDX板块成分股
- **Method**: GET
- **Path**: `/api/tdx/block_stocks`
- **Description**: 获取通达信板块成分股数据。支持按板块代码或股票代码筛选，如果不传参数则返回全部数据。

**Request Parameters (Query)**:
| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| block_code | string | 否 | 板块代码 (例如: 880506.TDX) |
| stock_code | string | 否 | 股票代码 (例如: 000063.SZ) |
| page | integer | 否 | 页码，默认 0 |
| page_size | integer | 否 | 每页条数，默认 10000 |

**Response**:
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 100,
    "page": 0,
    "page_size": 10000,
    "list": [
      {
        "block_code": "880506.TDX",
        "block_name": "5G概念",
        "block_type": 2,
        "stock_code": "000063.SZ"
      }
    ]
  }
}
```

##### 3.6.1.3 获取TDX板块日K
- **Method**: GET
- **Path**: `/api/tdx/daily`
- **Description**: 获取通达信板块指数日K线数据。支持按板块代码查询历史数据，或按日期查询当日所有板块数据。

**Request Parameters (Query)**:
| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| board_code | string | 否 | 板块代码 (例如: 880471或880471.TDX) |
| trade_date | string | 否 | 交易日期 (YYYY-MM-DD) |
| start_date | string | 否 | 开始日期 (YYYY-MM-DD) |
| end_date | string | 否 | 结束日期 (YYYY-MM-DD) |
| page | integer | 否 | 页码，默认 0 |
| page_size | integer | 否 | 每页条数，默认 100 |

**Response**:
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 100,
    "page": 0,
    "page_size": 100,
    "list": [
      {
        "board_code": "880471.TDX",
        "trade_date": "2024-02-26",
        "open": 1000.5,
        "high": 1010.2,
        "low": 998.0,
        "close": 1005.8,
        "vol": 50000000,
        "amount": 1000000000
      }
    ]
  }
}
```

##### 3.6.1.4 获取东财板块列表
- **Method**: POST
- **Path**: `/api/dc/blocks`
- **Description**: 获取东方财富板块列表数据，支持按板块代码、板块类型、板块名称筛选。

**Request Parameters (Body, JSON)**:
| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| block_code | string \| string[] | 否 | 板块代码 |
| block_type | string | 否 | 板块类型，例如：概念板块、行业板块、地域板块 |
| block_name | string | 否 | 板块名称（模糊匹配） |
| page | integer | 否 | 页码，默认 0 |
| page_size | integer | 否 | 每页条数，默认 2000，最大 8000 |

**Response**:
```json
{
  "code": 200,
  "msg": "成功",
  "data": {
    "total": 1,
    "list": [
      {
        "block_code": "BK0428.DC",
        "block_name": "绿色电力",
        "block_type": "概念板块",
        "level": ""
      }
    ]
  }
}
```

##### 3.6.1.5 获取东财板块成分股
- **Method**: POST
- **Path**: `/api/dc/block_stocks`
- **Description**: 获取东方财富板块成分股数据。支持按 `block_code`、`trade_date`、`stock_code` 任意组合筛选；如果三个条件都不传，默认返回最新交易日数据。

**Request Parameters (Body, JSON)**:
| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| block_code | string \| string[] | 否 | 板块代码 |
| trade_date | string | 否 | 交易日期，支持 YYYY-MM-DD 或 YYYYMMDD |
| stock_code | string \| string[] | 否 | 股票代码，例如 600000.SH |
| page | integer | 否 | 页码，默认 0 |
| page_size | integer | 否 | 每页条数，默认 2000，最大 8000 |

**Response**:
```json
{
  "code": 200,
  "msg": "成功",
  "data": {
    "total": 2,
    "list": [
      {
        "block_code": "BK0428.DC",
        "trade_date": "2026-04-24",
        "stock_code": "600905.SH",
        "stock_name": "三峡能源"
      },
      {
        "block_code": "BK0428.DC",
        "trade_date": "2026-04-24",
        "stock_code": "600863.SH",
        "stock_name": "内蒙华电"
      }
    ]
  }
}
```

##### 3.6.1.6 获取东财板块日K
- **Method**: POST
- **Path**: `/api/dc/daily`
- **Description**: 获取东方财富板块日K数据。`block_code` 和 `trade_date` 二选一即可，支持同时传入做交集筛选。

**Request Parameters (Body, JSON)**:
| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| block_code | string \| string[] | 否 | 板块代码 |
| trade_date | string | 否 | 交易日期，支持 YYYY-MM-DD 或 YYYYMMDD |
| page | integer | 否 | 页码，默认 0 |
| page_size | integer | 否 | 每页条数，默认 2000，最大 8000 |

> **注意**: `block_code` 与 `trade_date` 至少传一个。

**Response**:
```json
{
  "code": 200,
  "msg": "成功",
  "data": {
    "total": 1,
    "list": [
      {
        "block_code": "BK0428.DC",
        "trade_date": "2026-04-24",
        "open": 1512.4,
        "high": 1521.8,
        "low": 1498.2,
        "close": 1506.7,
        "change": -5.3,
        "pct_change": -0.35,
        "vol": 85612345,
        "amount": 1234567890.0,
        "swing": 1.56,
        "turnover_rate": 2.21
      }
    ]
  }
}
```

##### 3.6.1.7 获取历史分时
- **Method**: POST
- **Path**: `/api/index/history`
- **Description**: 获取指定指数的历史分时数据。

##### 3.6.1.8 获取同花顺板块分类
- **Method**: POST
- **Path**: `/api/index/ths_sector_categories`
- **Description**: 获取同花顺板块分类数据。

**Request Parameters (Body, JSON)**:
| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| type | string | 否 | 指数类型：N-概念指数 I-行业指数 R-地域指数 S-同花顺特色指数 ST-同花顺风格指数 TH-同花顺主题指数 BB-同花顺宽基指数 |
| page | integer | 否 | 页码，默认 0 |
| page_size | integer | 否 | 每页条数，默认 1000 |

**Response**:
```json
{
  "code": 200,
  "msg": "成功",
  "data": {
    "total": 1500,
    "list": [
      {
        "index_code": "882001.TI",
        "name": "安徽",
        "count": 159,
        "exchange": "A",
        "list_date": "20070808",
        "type": "R",
        "code": "882001"
      }
    ]
  }
}
```

##### 3.6.1.9 获取同花顺成分股
- **Method**: POST
- **Path**: `/api/index/ths_constituent_stocks`
- **Description**: 获取同花顺成分股数据。

**Request Parameters (Body, JSON)**:
| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| index_code | string | 否 | 指数代码，如 882001.TI |
| stock_code | string \| string[] | 否 | 股票代码 |
| page | integer | 否 | 页码，默认 0 |
| page_size | integer | 否 | 每页条数，默认 1000 |

**Response**:
```json
{
  "code": 200,
  "msg": "成功",
  "data": {
    "total": 35000,
    "list": [
      {
        "index_code": "882001.TI",
        "code": "000153",
        "stock_code": "000153.SZ",
        "stock_name": "丰原药业",
        "plate_name": "安徽"
      }
    ]
  }
}
```

##### 3.6.1.10 获取同花顺日线数据
- **Method**: POST
- **Path**: `/api/index/ths_daily`
- **Description**: 获取同花顺指数历史日K线数据。

**Request Parameters (Body, JSON)**:
| 字段名 | 类型 | 必填 | 说明 |
| :--- | :--- | :--- | :--- |
| ths_code | string \| string[] | 否 | 同花顺指数代码 (例如: 881123.TI) |
| start_time | string | 是 | 开始日期 (YYYY-MM-DD) |
| end_time | string | 是 | 结束日期 (YYYY-MM-DD) |
| page | integer | 否 | 页码，默认 0 |
| page_size | integer | 否 | 每页条数，默认 10000 |

**Response**:
```json
{
  "code": 200,
  "msg": "成功",
  "data": {
    "total": 100,
    "list": [
      {
        "ths_code": "881123.TI",
        "trade_date": "2026-04-10",
        "open": 1000.0,
        "high": 1010.0,
        "low": 990.0,
        "close": 1005.0,
        "pre_close": 1000.0,
        "avg_price": 1002.5,
        "change": 5.0,
        "pct_change": 0.5,
        "vol": 100000,
        "turnover_rate": 1.5
      }
    ]
  }
}
```

#### 3.6.2 实时数据 (Real-time Data)

##### 3.6.2.1 获取实时分时数据
- **Method**: POST
- **Path**: `/api/index/realtime/history`
- **Description**: 获取全市场或指定指数实时 1 分钟级别分时数据（支持查询最近7天内的数据）。返回数据会根据 index_code + trade_time 进行去重。注意：`index_code` 或 `trade_time` 至少提供一个。

**请求参数 (Request Parameters)**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `index_code` | string \| string[] | 否 | 指数代码，如 000001.SH 或 ["000001.SH", "399001.SZ"] (必须提供 index_code 或 trade_time 之一) |
| `trade_time` | string | 否 | 交易时间，如 2026-03-15 09:31:00 (必须提供 index_code 或 trade_time 之一) |
| `date` | string | 否 | 日期，格式 YYYY-MM-DD，默认今天，支持查询最近7天内的数据 |

**响应参数 (Response Parameters)**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `index_code` | string | 指数代码 |
| `trade_time` | string | 交易时间 |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价 |
| `vol` | float | 成交量(单位股) |
| `amount` | float | 成交额 |

**响应示例 (Response Example)**
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "date": "2026-03-15",
    "count": 1,
    "list": [
      {
        "index_code": "000001.SH",
        "trade_time": "2026-03-15 09:31:00",
        "open": 3000.5,
        "high": 3001.6,
        "low": 2999.4,
        "close": 3000.55,
        "vol": 10000,
        "amount": 105000
      }
    ]
  }
}
```

### 3.7 实时行情 (Real-time)

#### 3.7.0 当前涨停数据快照
- **Method**: POST
- **Path**: `/api/realtime/limit_up`
- **Description**: 获取涨停股票快照数据。默认返回全市场最新一次快照（最多10000条），传 `stock_code` 时返回该股票当天全部数据。

**请求参数 (Request Parameters)**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `date` | string | 否 | 日期，格式 YYYY-MM-DD，默认今天 |
| `stock_code` | string\|array | 否 | 股票代码，支持单个字符串或数组；传入后返回该股票（或多只股票）当天全部数据 |
| `page` | integer | 否 | 页码，从0开始，默认0 |
| `page_size` | integer | 否 | 每页数量，默认10000，最大10000（兼容 `pageSize`） |

**响应参数 (Response Parameters)**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 股票代码 |
| `update_time` | string | 更新时间 |
| `close` | float | 当前价 |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `pre_close` | float | 昨收价 |
| `vol` | integer | 成交量 |
| `amount` | float | 成交额 |
| `turnover_rate`| float | 换手率 |
| `ask_price`| float | 卖一价（跌停封单计算用） |
| `bid_price`| float | 买一价（涨停封单计算用） |
| `ask_vol`| integer | 卖一量（跌停封单计算用） |
| `bid_vol`| integer | 买一量（涨停封单计算用） |
| `volume_ratio`| float | 量比 |
| `pct_chg`| float | 涨跌幅(%) |
| `limit_status`| string | 涨跌停状态 |
| `seal_amount`| float | 封单金额（涨停=bid_price*bid_vol，跌停=ask_price*ask_vol） |

**响应示例 (Response Example)**
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "date": "2026-04-16",
    "count": 1,
    "list": [
      {
        "stock_code": "000001.SZ",
        "update_time": "2026-04-16 14:00:00",
        "insert_time": "2026-04-16 14:00:03",
        "close": 11.00,
        "open": 10.00,
        "high": 11.00,
        "low": 9.90,
        "pre_close": 10.00,
        "vol": 100000,
        "amount": 1050000,
        "turnover_rate": 1.5,
        "ask_price": 11.00,
        "bid_price": 11.00,
        "ask_vol": 0,
        "bid_vol": 52300,
        "volume_ratio": 1.2,
        "pct_chg": 10.0,
        "limit_status": "limit_up",
        "seal_amount": 575300.0
      }
    ]
  }
}
```

#### 3.7.1 全市场涨跌分布
- **Method**: POST
- **Path**: `/api/realtime/market_distribution`
- **Description**: 获取全市场分钟级别的涨跌家数、各档位分布及涨跌停数据。

**请求参数 (Request Parameters)**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `date` | string | 否 | 日期，格式 YYYY-MM-DD，默认今天 |

**响应参数 (Response Parameters)**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `trade_time` | string | 交易时间 |
| `up_count` | integer | 上涨家数 |
| `down_count` | integer | 下跌家数 |
| `flat_count` | integer | 平盘家数 |
| `limit_up_count` | integer | 涨停家数 |
| `limit_down_count` | integer | 跌停家数 |
| `up_over_10` | integer | 涨幅>10% |
| `up_7_to_10` | integer | 涨幅7%~10% |
| `up_5_to_7` | integer | 涨幅5%~7% |
| `up_3_to_5` | integer | 涨幅3%~5% |
| `up_0_to_3` | integer | 涨幅0%~3% |
| `zero` | integer | 涨幅0% |
| `down_0_to_3` | integer | 跌幅0%~-3% |
| `down_3_to_5` | integer | 跌幅-3%~-5% |
| `down_5_to_7` | integer | 跌幅-5%~-7% |
| `down_7_to_10` | integer | 跌幅-7%~-10% |
| `down_over_10` | integer | 跌幅<-10% |

**响应示例 (Response Example)**
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "date": "2026-04-16",
    "count": 1,
    "list": [
      {
        "trade_time": "2026-04-16 09:30:00",
        "up_count": 3000,
        "down_count": 2000,
        "flat_count": 300,
        "limit_up_count": 50,
        "limit_down_count": 10,
        "up_over_10": 20,
        "up_7_to_10": 80,
        "up_5_to_7": 200,
        "up_3_to_5": 500,
        "up_0_to_3": 2200,
        "zero": 300,
        "down_0_to_3": 1500,
        "down_3_to_5": 300,
        "down_5_to_7": 100,
        "down_7_to_10": 80,
        "down_over_10": 20
      }
    ]
  }
}
```

#### 3.7.2 竞价快照数据
- **Method**: POST
- **Path**: `/api/realtime/auction_daily`
- **Description**: 获取当天竞价时间 (09:15-09:25) 的快照数据，包括最新价、成交量等竞价信息。支持按时间区间查询历史快照序列。

**请求参数 (Request Parameters)**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string\|array | 否 | 股票代码，支持单个字符串或数组 |
| `date` | string | 否 | 日期，格式 YYYY-MM-DD，默认今天。如果不传时间区间，则返回该日期内每只股票最新的一条。 |
| `start_time` | string | 否 | 开始时间，格式 YYYY-MM-DD HH:MM:SS，如果传了该参数，则查询指定时间区间的历史序列数据 |
| `end_time` | string | 否 | 结束时间，格式 YYYY-MM-DD HH:MM:SS |
| `page` | integer | 否 | 页码，从0开始 |
| `page_size` | integer | 否 | 每页数量 |

**响应参数 (Response Parameters)**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 股票代码 |
| `stock_name` | string | 股票名称 |
| `update_time` | string | 更新时间 |
| `close` | float | 当前价/收盘价 |
| `pre_close` | float | 昨收价 |
| `vol` | integer | 成交量 |
| `amount` | float | 成交额 |
| `turnover_rate`| float | 换手率 |
| `ask_price` | float | 卖一价 |
| `bid_price` | float | 买一价 |
| `ask_vol` | integer | 卖一量 |
| `bid_vol` | integer | 买一量 |

#### 3.7.2 WebSocket 快照推送
- **URL**: `wss://data.diemeng.chat/ws/stock/snapshot`
- **鉴权**: URL参数 `token`
- **Description**: 实时推送全市场股票的快照数据。

**数据格式**: 二进制 Gzip 压缩的 JSON 数组。
**字段映射**:
- `cd`: Code (股票代码)
- `lp`: Last Price (最新价)
- `vo`: Volume (成交量)
- `to`: Turnover (成交额)

**Python 示例**:
```python
import websocket
import gzip
import json

def on_message(ws, message):
    data = json.loads(gzip.decompress(message).decode('utf-8'))
    print(data)

ws = websocket.WebSocketApp("wss://data.diemeng.chat/ws/stock/snapshot?token=YOUR_API_KEY",
                            on_message=on_message)
ws.run_forever()
```

#### 3.7.3 涨跌停状态 WS 推送
- **URL**: `wss://data.diemeng.chat/ws/stream`
- **鉴权**: URL参数 `token`
- **订阅**: URL参数 `types`（逗号分隔）
- **Description**: 订阅涨跌停状态变化事件与持续更新事件。

**连接示例**
- `wss://data.diemeng.chat/ws/stream?token=YOUR_API_KEY&types=limit_up_update,limit_down_update`

**动态订阅示例**
```json
{"action":"subscribe","types":["limit_up_update","limit_down_update"]}
```

**推送消息结构**
```json
{
  "type": "stock_limit_event",
  "data": {
    "type": "limit_up_update",
    "stock_code": "600000.SH",
    "stock_name": "浦发银行",
    "last_price": 10.5,
    "high_price": 10.5,
    "low_price": 10.2,
    "volume": 123456789,
    "turnover": 1234567890.12,
    "bid_price": 10.5,
    "bid_vol": 180000,
    "ask_price": 0,
    "ask_vol": 0,
    "seal_price": 10.5,
    "seal_vol": 180000,
    "change_rate": 10.0,
    "source": "fast",
    "timestamp": "2026-04-17 11:23:45"
  }
}
```

**`limit_up_update` / `limit_down_update` 字段说明**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `type` | string | 外层固定为 `stock_limit_event` |
| `data.type` | string | 事件类型：`limit_up_update` / `limit_down_update` |
| `data.stock_code` | string | 股票代码 |
| `data.stock_name` | string | 股票名称 |
| `data.last_price` | float | 最新价 |
| `data.high_price` | float | 最高价 |
| `data.low_price` | float | 最低价 |
| `data.volume` | int64 | 成交量 |
| `data.turnover` | float | 成交额 |
| `data.bid_price` | float | 买一价 |
| `data.bid_vol` | int64 | 买一量 |
| `data.ask_price` | float | 卖一价 |
| `data.ask_vol` | int64 | 卖一量 |
| `data.seal_price` | float | 封单价（涨停取买一，跌停取卖一） |
| `data.seal_vol` | int64 | 封单量（涨停取买一量，跌停取卖一量） |
| `data.change_rate` | float | 涨跌幅（%） |
| `data.source` | string | 来源 worker（fast/normal） |
| `data.timestamp` | string | 服务端推送时间 |

#### 3.7.4 获取实时分时数据
- **Method**: POST
- **Path**: `/api/realtime/history`
- **Description**: 获取全市场或指定股票实时，包括股票，指数，可转债 1 分钟级别分时数据（支持查询最近7天内的数据）。返回数据会根据 stock_code + trade_time 进行去重。注意：`stock_code` 或 `trade_time` 至少提供一个。

**请求参数 (Request Parameters)**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `stock_code` | string \| string[] | 否 | 股票代码，如 600000.SH 或 ["600000.SH", "000001.SZ"] (必须提供 stock_code 或 trade_time 之一) |
| `trade_time` | string | 否 | 交易时间，如 2026-03-15 09:31:00 (必须提供 stock_code 或 trade_time 之一) |
| `date` | string | 否 | 日期，格式 YYYY-MM-DD，默认今天，支持查询最近7天内的数据 |

**响应参数 (Response Parameters)**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `stock_code` | string | 股票代码 |
| `trade_time` | string | 交易时间 |
| `open` | float | 开盘价 |
| `high` | float | 最高价 |
| `low` | float | 最低价 |
| `close` | float | 收盘价 |
| `vol` | float | 成交量 |
| `amount` | float | 成交额 |

**响应示例 (Response Example)**
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "date": "2026-03-15",
    "count": 1,
    "list": [
      {
        "stock_code": "600000.SH",
        "trade_time": "2026-03-15 09:31:00",
        "open": 10.5,
        "high": 10.6,
        "low": 10.4,
        "close": 10.55,
        "vol": 10000,
        "amount": 105000
      }
    ]
  }
}
```

#### 3.7.10 龙虎榜机构明细 (Dragon Tiger List)
- **Method**: POST
- **Path**: `/api/stock/dragon_tiger`
- **Description**: 查询龙虎榜机构明细数据。`date` 和 `stock_code` 必须至少提供一个。

**请求参数 (Request Parameters)**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `date` | string | 否 | 交易日期 (YYYY-MM-DD) |
| `stock_code` | string | 否 | 股票代码 (e.g. 600519.SH) |
| `page` | integer | 否 | 页码 (默认 1) |
| `page_size` | integer | 否 | 每页数量 (默认 20, 最大 1000) |

**响应参数 (Response Parameters)**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `trade_date` | string | 交易日期 |
| `stock_code` | string | 股票代码 |
| `org_name` | string | 营业部名称 |
| `buy_amount` | float | 买入额(万) |
| `buy_ratio` | float | 买入占比 |
| `sell_amount` | float | 卖出额(万) |
| `sell_ratio` | float | 卖出占比 |
| `net_buy_amount` | float | 净买入额(万) |
| `direction` | int | 买卖方向 (0:未知, 1:买入, 2:卖出) |
| `reason` | string | 上榜理由 |

**响应示例 (Response Example)**
```json
{
    "code": 200,
    "msg": "Success",
    "data": [
        {
            "trade_date": "2026-03-16",
            "stock_code": "000533.SZ",
            "org_name": "东方财富证券股份有限公司拉萨团结路第一证券营业部",
            "buy_amount": 1868.90,
            "buy_ratio": 0.52,
            "sell_amount": 3028.64,
            "sell_ratio": 0.84,
            "net_buy_amount": -1159.74,
            "direction": 1,
            "reason": "日振幅值达到15%的前5只证券"
        }
    ],
    "total": 10
}
```

#### 3.7.11 龙虎榜每日明细 (Top List)
- **Method**: POST
- **Path**: `/api/stock/top_list`
- **Description**: 查询龙虎榜每日交易明细数据。

**请求参数 (Request Parameters)**
| 参数名 | 类型 | 必选 | 说明 |
| :--- | :--- | :--- | :--- |
| `trade_date` | string | 是 | 交易日期 (YYYY-MM-DD) |
| `stock_code` | string | 否 | 股票代码 (e.g. 600519.SH) |

**响应参数 (Response Parameters)**
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `trade_date` | string | 交易日期 |
| `stock_code` | string | 股票代码 |
| `name` | string | 名称 |
| `close` | float | 收盘价 |
| `pct_change` | float | 涨跌幅 |
| `turnover_rate` | float | 换手率 |
| `amount` | float | 总成交额 |
| `l_sell` | float | 龙虎榜卖出额 |
| `l_buy` | float | 龙虎榜买入额 |
| `l_amount` | float | 龙虎榜成交额 |
| `net_amount` | float | 龙虎榜净买入额 |
| `net_rate` | float | 龙虎榜净买额占比 |
| `amount_rate` | float | 龙虎榜成交额占比 |
| `float_values` | float | 当日流通市值 |
| `reason` | string | 上榜理由 |

**响应示例 (Response Example)**
```json
{
    "code": 200,
    "msg": "Success",
    "data": [
        {
            "trade_date": "2026-03-16",
            "stock_code": "000533.SZ",
            "name": "万家乐",
            "close": 10.5,
            "pct_change": 10.01,
            "turnover_rate": 15.2,
            "amount": 15000.5,
            "l_sell": 5000.2,
            "l_buy": 8000.5,
            "l_amount": 13000.7,
            "net_amount": 3000.3,
            "net_rate": 20.5,
            "amount_rate": 85.2,
            "float_values": 50000.5,
            "reason": "日涨幅偏离值达到7%的前5只证券"
        }
    ]
}
```

---

## 4. WebSocket 接口 (WebSocket Interface)
```
