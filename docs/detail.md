接口名称: API 使用指南 (Usage Guide)
接口地址: /api

接口说明:
**API 使用指南**

**1. 基础地址 (Base URL)**
https://data.diemeng.chat/api

**2. 鉴权方式 (Authentication)**
所有接口都需要进行鉴权。请在请求头 (Header) 中携带 `apiKey`。
- **Header Key**: `apiKey` (推荐) 或 `X-API-Key`
- **Value**: 您的 API 密钥 (可在个人中心获取)

**示例 (Curl)**:
`curl -H "apiKey: your_api_key" https://data.diemeng.chat/api/stock/list`

**3. 通用返回参数**
所有接口均返回 JSON 格式数据，包含以下通用字段：
- `code` (int): 状态码。200 表示成功，其他表示失败。
- `msg` (string): 提示信息。成功时为 "Success"，失败时为错误描述。
- `data` (object/array): 具体的业务数据。

**4. 常见错误码**
- 401: 未授权 (Missing or invalid apiKey)
- 403: 权限不足 (Insufficient permissions)
- 429: 请求过于频繁 (Rate limit exceeded)
- 500: 服务器内部错误

响应示例:
```json
{
  "code": 200,
  "msg": "Success",
  "data": {}
}
```
接口名称: Python 调用示例 (Python Examples)
接口地址: /api/examples/python

接口说明:
**Python 调用示例**

本节提供使用 Python `requests` 库调用核心接口的示例代码。

**1. 准备工作**
请确保已安装 requests 库：
```bash
pip install requests
```

**2. 获取 API Key**
- 登录本平台
- 进入 **个人中心 (User Center)**
- 在 **API 管理** 或 **密钥管理** 栏目中复制您的 `apiKey`

**3. 示例代码**

### 3.1 获取股票列表 (Get Stock List)
```python
import requests

url = "https://data.diemeng.chat/api/stock/list"
headers = {
    "apiKey": "YOUR_API_KEY"  # 替换为您的真实 Key
}

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    result = response.json()
    
    if result["code"] == 200:
        print("股票列表获取成功:")
        # 注意: 返回数据包含 total 和 list
        stock_list = result["data"]["list"]
        for stock in stock_list[:5]:  # 打印前5条
            print(stock)
    else:
        print(f"Error: {result['msg']}")
except Exception as e:
    print(f"Request Failed: {e}")
```

**Response Example (JSON):**
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 5300,
    "list": [
      {
        "stock_code": "600000.SH",
        "name": "浦发银行",
        "area": "上海",
        "industry": "银行",
        "market": "主板",
        "list_date": "1999-11-10",
        "symbol": "600000",
        "act_name": "上海浦东发展银行股份有限公司",
        "act_ent_type": "股份有限公司",
        "list_status": "L",
        "delist_date": null,
        "is_hs": "H"
      }
    ]
  }
}
```

### 3.2 获取日K线数据 (Get Daily Data)
```python
import requests

url = "https://data.diemeng.chat/api/stock/daily"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "start_time": "2023-01-01",
    "end_time": "2023-01-10",
    "page": 0,
    "page_size": 100
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

**Response Example (JSON):**
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 100,
    "list": [
      {
        "stock_code": "600000.SH",
        "stock_name": "浦发银行",
        "trade_date": "2023-01-05",
        "open": 7.35,
        "high": 7.39,
        "low": 7.33,
        "close": 7.37,
        "pre_close": 7.35,
        "change": 0.02,
        "pct_chg": 0.27,
        "vol": 200000.0,
        "amount": 1470000.0
      }
    ]
  }
}
```

### 3.3 获取每日财务数据 (Get Finance Data)
```python
import requests

url = "https://data.diemeng.chat/api/stock/finance"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "start_time": "2023-01-01",
    "end_time": "2023-01-31"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

**Response Example (JSON):**
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 20,
    "list": [
      {
        "stock_code": "600000.SH",
        "trade_date": "2023-01-05",
        "close": 7.37,
        "turnover_rate": 0.5,
        "turnover_rate_f": 0.5,
        "volume_ratio": 1.1,
        "pe": 4.5,
        "pe_ttm": 4.2,
        "pb": 0.5,
        "ps": 1.2,
        "ps_ttm": 1.1,
        "dv_ratio": 3.5,
        "dv_ttm": 3.4,
        "total_share": 29352000000.0,
        "float_share": 29352000000.0,
        "free_share": 28000000000.0,
        "total_mv": 216300000000.0,
        "circ_mv": 216300000000.0
      }
    ]
  }
}
```


响应示例:
```json
See code examples above
```
接口名称: 获取交易日历
请求方式: GET/POST
接口地址: /api/basic/calendar

接口说明:
获取交易日历数据。支持按日期范围查询。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| start_time | string | 是 | 开始日期 (YYYY-MM-DD) |
| end_time | string | 是 | 结束日期 (YYYY-MM-DD) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| date | string | 日历日期 |
| is_open | integer | 是否交易日 (0:休市, 1:交易) |

请求示例 (Python):
```python
# 暂无示例
```

响应示例:
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
接口名称: 获取财务指标报表数据
请求方式: POST
接口地址: /api/stock/financial_indicator

接口说明:
从 stock_financial_indicator 表获取股票财务指标数据。请求参数 stock_code / end_date / ann_date 三选一至少传一个。每次最多返回 10000 条数据。接口实际返回该表全部字段（除 update_flag、create_time），下方已展示完整字段清单。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| stock_code | string | string[] | 否 | 股票代码，例如 "600000.SH"。支持数组。 |
| end_date | string | 否 | 报告期最后日期，格式 YYYY-MM-DD。 |
| ann_date | string | 否 | 公告日期，格式 YYYY-MM-DD。 |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000，最大10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| stock_code | string | 股票代码（对应 Tushare 的 ts_code） |
| ann_date | string | 公告日期 |
| end_date | string | 报告期 |
| eps | float | string | null | 基本每股收益 |
| dt_eps | float | string | null | 稀释每股收益 |
| total_revenue_ps | float | string | null | 每股营业总收入 |
| revenue_ps | float | string | null | 每股营业收入 |
| capital_rese_ps | float | string | null | 每股资本公积 |
| surplus_rese_ps | float | string | null | 每股盈余公积 |
| undist_profit_ps | float | string | null | 每股未分配利润 |
| extra_item | float | string | null | 非经常性损益 |
| profit_dedt | float | string | null | 扣除非经常性损益后的净利润（扣非净利润） |
| gross_margin | float | string | null | 毛利 |
| current_ratio | float | string | null | 流动比率 |
| quick_ratio | float | string | null | 速动比率 |
| cash_ratio | float | string | null | 保守速动比率 |
| invturn_days | float | string | null | 存货周转天数 |
| arturn_days | float | string | null | 应收账款周转天数 |
| inv_turn | float | string | null | 存货周转率 |
| ar_turn | float | string | null | 应收账款周转率 |
| ca_turn | float | string | null | 流动资产周转率 |
| fa_turn | float | string | null | 固定资产周转率 |
| assets_turn | float | string | null | 总资产周转率 |
| op_income | float | string | null | 经营活动净收益 |
| valuechange_income | float | string | null | 价值变动净收益 |
| interst_income | float | string | null | 利息费用 |
| daa | float | string | null | 折旧与摊销 |
| ebit | float | string | null | 息税前利润 |
| ebitda | float | string | null | 息税折旧摊销前利润 |
| fcff | float | string | null | 企业自由现金流量 |
| fcfe | float | string | null | 股权自由现金流量 |
| current_exint | float | string | null | 无息流动负债 |
| noncurrent_exint | float | string | null | 无息非流动负债 |
| interestdebt | float | string | null | 带息债务 |
| netdebt | float | string | null | 净债务 |
| tangible_asset | float | string | null | 有形资产 |
| working_capital | float | string | null | 营运资金 |
| networking_capital | float | string | null | 营运流动资本 |
| invest_capital | float | string | null | 全部投入资本 |
| retained_earnings | float | string | null | 留存收益 |
| diluted2_eps | float | string | null | 期末摊薄每股收益 |
| bps | float | string | null | 每股净资产 |
| ocfps | float | string | null | 每股经营活动产生的现金流量净额 |
| retainedps | float | string | null | 每股留存收益 |
| cfps | float | string | null | 每股现金流量净额 |
| ebit_ps | float | string | null | 每股息税前利润 |
| fcff_ps | float | string | null | 每股企业自由现金流量 |
| fcfe_ps | float | string | null | 每股股东自由现金流量 |
| netprofit_margin | float | string | null | 销售净利率 |
| grossprofit_margin | float | string | null | 销售毛利率 |
| cogs_of_sales | float | string | null | 销售成本率 |
| expense_of_sales | float | string | null | 销售期间费用率 |
| profit_to_gr | float | string | null | 净利润/营业总收入 |
| saleexp_to_gr | float | string | null | 销售费用/营业总收入 |
| adminexp_of_gr | float | string | null | 管理费用/营业总收入 |
| finaexp_of_gr | float | string | null | 财务费用/营业总收入 |
| impai_ttm | float | string | null | 资产减值损失/营业总收入 |
| gc_of_gr | float | string | null | 营业总成本/营业总收入 |
| op_of_gr | float | string | null | 营业利润/营业总收入 |
| ebit_of_gr | float | string | null | 息税前利润/营业总收入 |
| roe | float | string | null | 净资产收益率 |
| roe_waa | float | string | null | 加权平均净资产收益率 |
| roe_dt | float | string | null | 净资产收益率(扣除非经常损益) |
| roa | float | string | null | 总资产报酬率 |
| npta | float | string | null | 总资产净利润 |
| roic | float | string | null | 投入资本回报率 |
| roe_yearly | float | string | null | 年化净资产收益率 |
| roa2_yearly | float | string | null | 年化总资产报酬率 |
| roe_avg | float | string | null | 平均净资产收益率(增发条件) |
| opincome_of_ebt | float | string | null | 经营活动净收益/利润总额 |
| investincome_of_ebt | float | string | null | 价值变动净收益/利润总额 |
| n_op_profit_of_ebt | float | string | null | 营业外收支净额/利润总额 |
| tax_to_ebt | float | string | null | 所得税/利润总额 |
| dtprofit_to_profit | float | string | null | 扣除非经常损益后的净利润/净利润 |
| salescash_to_or | float | string | null | 销售商品提供劳务收到的现金/营业收入 |
| ocf_to_or | float | string | null | 经营活动产生的现金流量净额/营业收入 |
| ocf_to_opincome | float | string | null | 经营活动产生的现金流量净额/经营活动净收益 |
| capitalized_to_da | float | string | null | 资本支出/折旧和摊销 |
| debt_to_assets | float | string | null | 资产负债率 |
| assets_to_eqt | float | string | null | 权益乘数 |
| dp_assets_to_eqt | float | string | null | 权益乘数(杜邦分析) |
| ca_to_assets | float | string | null | 流动资产/总资产 |
| nca_to_assets | float | string | null | 非流动资产/总资产 |
| tbassets_to_totalassets | float | string | null | 有形资产/总资产 |
| int_to_talcap | float | string | null | 带息债务/全部投入资本 |
| eqt_to_talcapital | float | string | null | 归属于母公司的股东权益/全部投入资本 |
| currentdebt_to_debt | float | string | null | 流动负债/负债合计 |
| longdeb_to_debt | float | string | null | 非流动负债/负债合计 |
| ocf_to_shortdebt | float | string | null | 经营活动产生的现金流量净额/流动负债 |
| debt_to_eqt | float | string | null | 产权比率 |
| eqt_to_debt | float | string | null | 归属于母公司的股东权益/负债合计 |
| eqt_to_interestdebt | float | string | null | 归属于母公司的股东权益/带息债务 |
| tangibleasset_to_debt | float | string | null | 有形资产/负债合计 |
| tangasset_to_intdebt | float | string | null | 有形资产/带息债务 |
| tangibleasset_to_netdebt | float | string | null | 有形资产/净债务 |
| ocf_to_debt | float | string | null | 经营活动产生的现金流量净额/负债合计 |
| ocf_to_interestdebt | float | string | null | 经营活动产生的现金流量净额/带息债务 |
| ocf_to_netdebt | float | string | null | 经营活动产生的现金流量净额/净债务 |
| ebit_to_interest | float | string | null | 已获利息倍数(EBIT/利息费用) |
| longdebt_to_workingcapital | float | string | null | 长期债务与营运资金比率 |
| ebitda_to_debt | float | string | null | 息税折旧摊销前利润/负债合计 |
| turn_days | float | string | null | 营业周期 |
| roa_yearly | float | string | null | 年化总资产净利率 |
| roa_dp | float | string | null | 总资产净利率(杜邦分析) |
| fixed_assets | float | string | null | 固定资产合计 |
| profit_prefin_exp | float | string | null | 扣除财务费用前营业利润 |
| non_op_profit | float | string | null | 非营业利润 |
| op_to_ebt | float | string | null | 营业利润／利润总额 |
| nop_to_ebt | float | string | null | 非营业利润／利润总额 |
| ocf_to_profit | float | string | null | 经营活动产生的现金流量净额／营业利润 |
| cash_to_liqdebt | float | string | null | 货币资金／流动负债 |
| cash_to_liqdebt_withinterest | float | string | null | 货币资金／带息流动负债 |
| op_to_liqdebt | float | string | null | 营业利润／流动负债 |
| op_to_debt | float | string | null | 营业利润／负债合计 |
| roic_yearly | float | string | null | 年化投入资本回报率 |
| total_fa_trun | float | string | null | 固定资产合计周转率 |
| profit_to_op | float | string | null | 利润总额／营业收入 |
| q_opincome | float | string | null | 经营活动单季度净收益 |
| q_investincome | float | string | null | 价值变动单季度净收益 |
| q_dtprofit | float | string | null | 扣除非经常损益后的单季度净利润 |
| q_eps | float | string | null | 每股收益(单季度) |
| q_netprofit_margin | float | string | null | 销售净利率(单季度) |
| q_gsprofit_margin | float | string | null | 销售毛利率(单季度) |
| q_exp_to_sales | float | string | null | 销售期间费用率(单季度) |
| q_profit_to_gr | float | string | null | 净利润／营业总收入(单季度) |
| q_saleexp_to_gr | float | string | null | 销售费用／营业总收入(单季度) |
| q_adminexp_to_gr | float | string | null | 管理费用／营业总收入(单季度) |
| q_finaexp_to_gr | float | string | null | 财务费用／营业总收入(单季度) |
| q_impair_to_gr_ttm | float | string | null | 资产减值损失／营业总收入(单季度) |
| q_gc_to_gr | float | string | null | 营业总成本／营业总收入(单季度) |
| q_op_to_gr | float | string | null | 营业利润／营业总收入(单季度) |
| q_roe | float | string | null | 净资产收益率(单季度) |
| q_dt_roe | float | string | null | 净资产单季度收益率(扣除非经常损益) |
| q_npta | float | string | null | 总资产净利润(单季度) |
| q_opincome_to_ebt | float | string | null | 经营活动净收益／利润总额(单季度) |
| q_investincome_to_ebt | float | string | null | 价值变动净收益／利润总额(单季度) |
| q_dtprofit_to_profit | float | string | null | 扣非净利润／净利润(单季度) |
| q_salescash_to_or | float | string | null | 销售商品提供劳务收到的现金／营业收入(单季度) |
| q_ocf_to_sales | float | string | null | 经营活动现金流净额／营业收入(单季度) |
| q_ocf_to_or | float | string | null | 经营活动现金流净额／经营活动净收益(单季度) |
| basic_eps_yoy | float | string | null | 基本每股收益同比增长率(%) |
| dt_eps_yoy | float | string | null | 稀释每股收益同比增长率(%) |
| cfps_yoy | float | string | null | 每股经营活动现金流净额同比增长率(%) |
| op_yoy | float | string | null | 营业利润同比增长率(%) |
| ebt_yoy | float | string | null | 利润总额同比增长率(%) |
| netprofit_yoy | float | string | null | 归母净利润同比增长率(%) |
| dt_netprofit_yoy | float | string | null | 归母净利润(扣非)同比增长率(%) |
| ocf_yoy | float | string | null | 经营活动现金流净额同比增长率(%) |
| roe_yoy | float | string | null | 净资产收益率(摊薄)同比增长率(%) |
| bps_yoy | float | string | null | 每股净资产相对年初增长率(%) |
| assets_yoy | float | string | null | 资产总计相对年初增长率(%) |
| eqt_yoy | float | string | null | 归母股东权益相对年初增长率(%) |
| tr_yoy | float | string | null | 营业总收入同比增长率(%) |
| or_yoy | float | string | null | 营业收入同比增长率(%) |
| q_gr_yoy | float | string | null | 营业总收入同比增长率(%)(单季度) |
| q_gr_qoq | float | string | null | 营业总收入环比增长率(%)(单季度) |
| q_sales_yoy | float | string | null | 营业收入同比增长率(%)(单季度) |
| q_sales_qoq | float | string | null | 营业收入环比增长率(%)(单季度) |
| q_op_yoy | float | string | null | 营业利润同比增长率(%)(单季度) |
| q_op_qoq | float | string | null | 营业利润环比增长率(%)(单季度) |
| q_profit_yoy | float | string | null | 净利润同比增长率(%)(单季度) |
| q_profit_qoq | float | string | null | 净利润环比增长率(%)(单季度) |
| q_netprofit_yoy | float | string | null | 归母净利润同比增长率(%)(单季度) |
| q_netprofit_qoq | float | string | null | 归母净利润环比增长率(%)(单季度) |
| equity_yoy | float | string | null | 净资产同比增长率 |
| rd_exp | float | string | null | 研发费用 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/financial_indicator"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "end_date": "2026-05-11",
    "ann_date": "2026-05-11",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 2,
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
接口名称: 获取利润表数据
请求方式: POST
接口地址: /api/stock/income

接口说明:
从 stock_income 表获取利润表数据。请求参数 stock_code / end_date / ann_date 三选一至少传一个。每次最多返回 10000 条数据。接口实际返回该表全部字段（除 update_flag、create_time），下方已展示完整字段清单。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| stock_code | string | string[] | 否 | 股票代码，例如 "600000.SH"。支持数组。 |
| end_date | string | 否 | 报告期最后日期，格式 YYYY-MM-DD。 |
| ann_date | string | 否 | 公告日期，格式 YYYY-MM-DD。 |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000，最大10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| stock_code | string | 股票代码（对应 Tushare 的 ts_code） |
| ann_date | string | 公告日期 |
| f_ann_date | string | 实际公告日期 |
| end_date | string | 报告期 |
| report_type | float | string | null | 报表类型 |
| comp_type | float | string | null | 公司类型(1一般工商业2银行3保险4证券) |
| end_type | float | string | null | 报告期类型 |
| basic_eps | float | string | null | 基本每股收益 |
| diluted_eps | float | string | null | 稀释每股收益 |
| total_revenue | float | string | null | 营业总收入 |
| revenue | float | string | null | 营业收入 |
| int_income | float | string | null | 利息收入 |
| prem_earned | float | string | null | 已赚保费 |
| comm_income | float | string | null | 手续费及佣金收入 |
| n_commis_income | float | string | null | 手续费及佣金净收入 |
| n_oth_income | float | string | null | 其他经营净收益 |
| n_oth_b_income | float | string | null | 加:其他业务净收益 |
| prem_income | float | string | null | 保险业务收入 |
| out_prem | float | string | null | 减:分出保费 |
| une_prem_reser | float | string | null | 提取未到期责任准备金 |
| reins_income | float | string | null | 其中:分保费收入 |
| n_sec_tb_income | float | string | null | 代理买卖证券业务净收入 |
| n_sec_uw_income | float | string | null | 证券承销业务净收入 |
| n_asset_mg_income | float | string | null | 受托客户资产管理业务净收入 |
| oth_b_income | float | string | null | 其他业务收入 |
| fv_value_chg_gain | float | string | null | 加:公允价值变动净收益 |
| invest_income | float | string | null | 加:投资净收益 |
| ass_invest_income | float | string | null | 其中:对联营企业和合营企业的投资收益 |
| forex_gain | float | string | null | 加:汇兑净收益 |
| total_cogs | float | string | null | 营业总成本 |
| oper_cost | float | string | null | 减:营业成本 |
| int_exp | float | string | null | 减:利息支出 |
| comm_exp | float | string | null | 减:手续费及佣金支出 |
| biz_tax_surchg | float | string | null | 减:营业税金及附加 |
| sell_exp | float | string | null | 减:销售费用 |
| admin_exp | float | string | null | 减:管理费用 |
| fin_exp | float | string | null | 减:财务费用 |
| assets_impair_loss | float | string | null | 减:资产减值损失 |
| prem_refund | float | string | null | 退保金 |
| compens_payout | float | string | null | 赔付总支出 |
| reser_insur_liab | float | string | null | 提取保险责任准备金 |
| div_payt | float | string | null | 保户红利支出 |
| reins_exp | float | string | null | 分保费用 |
| oper_exp | float | string | null | 营业支出 |
| compens_payout_refu | float | string | null | 减:摊回赔付支出 |
| insur_reser_refu | float | string | null | 减:摊回保险责任准备金 |
| reins_cost_refund | float | string | null | 减:摊回分保费用 |
| other_bus_cost | float | string | null | 其他业务成本 |
| operate_profit | float | string | null | 营业利润 |
| non_oper_income | float | string | null | 加:营业外收入 |
| non_oper_exp | float | string | null | 减:营业外支出 |
| nca_disploss | float | string | null | 其中:减:非流动资产处置净损失 |
| total_profit | float | string | null | 利润总额 |
| income_tax | float | string | null | 所得税费用 |
| n_income | float | string | null | 净利润(含少数股东损益) |
| n_income_attr_p | float | string | null | 净利润(不含少数股东损益) |
| minority_gain | float | string | null | 少数股东损益 |
| oth_compr_income | float | string | null | 其他综合收益 |
| t_compr_income | float | string | null | 综合收益总额 |
| compr_inc_attr_p | float | string | null | 归属于母公司(或股东)的综合收益总额 |
| compr_inc_attr_m_s | float | string | null | 归属于少数股东的综合收益总额 |
| ebit | float | string | null | 息税前利润 |
| ebitda | float | string | null | 息税折旧摊销前利润 |
| insurance_exp | float | string | null | 保险业务支出 |
| undist_profit | float | string | null | 年初未分配利润 |
| distable_profit | float | string | null | 可分配利润 |
| rd_exp | float | string | null | 研发费用 |
| fin_exp_int_exp | float | string | null | 财务费用:利息费用 |
| fin_exp_int_inc | float | string | null | 财务费用:利息收入 |
| transfer_surplus_rese | float | string | null | 盈余公积转入 |
| transfer_housing_imprest | float | string | null | 住房周转金转入 |
| transfer_oth | float | string | null | 其他转入 |
| adj_lossgain | float | string | null | 调整以前年度损益 |
| withdra_legal_surplus | float | string | null | 提取法定盈余公积 |
| withdra_legal_pubfund | float | string | null | 提取法定公益金 |
| withdra_biz_devfund | float | string | null | 提取企业发展基金 |
| withdra_rese_fund | float | string | null | 提取储备基金 |
| withdra_oth_ersu | float | string | null | 提取任意盈余公积金 |
| workers_welfare | float | string | null | 职工奖金福利 |
| distr_profit_shrhder | float | string | null | 可供股东分配的利润 |
| prfshare_payable_dvd | float | string | null | 应付优先股股利 |
| comshare_payable_dvd | float | string | null | 应付普通股股利 |
| capit_comstock_div | float | string | null | 转作股本的普通股股利 |
| continued_net_profit | float | string | null | 持续经营净利润 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/income"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "end_date": "2026-05-11",
    "ann_date": "2026-05-11",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
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
        "total_revenue": 123456000000,
        "revenue": 118234000000,
        "operate_profit": 32500000000,
        "total_profit": 33120000000,
        "n_income": 26890000000,
        "n_income_attr_p": 26120000000,
        "basic_eps": 1.24,
        "diluted_eps": 1.22
      }
    ]
  }
}
```
接口名称: 获取资产负债表数据
请求方式: POST
接口地址: /api/stock/balancesheet

接口说明:
从 stock_balancesheet 表获取资产负债表数据。请求参数 stock_code / end_date / ann_date 三选一至少传一个。每次最多返回 10000 条数据。接口实际返回该表全部字段（除 update_flag、create_time），下方已展示完整字段清单。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| stock_code | string | string[] | 否 | 股票代码，例如 "600000.SH"。支持数组。 |
| end_date | string | 否 | 报告期最后日期，格式 YYYY-MM-DD。 |
| ann_date | string | 否 | 公告日期，格式 YYYY-MM-DD。 |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000，最大10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| stock_code | string | 股票代码（对应 Tushare 的 ts_code） |
| ann_date | string | 公告日期 |
| f_ann_date | string | 实际公告日期 |
| end_date | string | 报告期 |
| report_type | float | string | null | 报表类型 |
| comp_type | float | string | null | 公司类型(1一般工商业2银行3保险4证券) |
| end_type | float | string | null | 报告期类型 |
| total_share | float | string | null | 期末总股本 |
| cap_rese | float | string | null | 资本公积金 |
| undistr_porfit | float | string | null | 未分配利润 |
| surplus_rese | float | string | null | 盈余公积金 |
| special_rese | float | string | null | 专项储备 |
| money_cap | float | string | null | 货币资金 |
| trad_asset | float | string | null | 交易性金融资产 |
| notes_receiv | float | string | null | 应收票据 |
| accounts_receiv | float | string | null | 应收账款 |
| oth_receiv | float | string | null | 其他应收款 |
| prepayment | float | string | null | 预付款项 |
| div_receiv | float | string | null | 应收股利 |
| int_receiv | float | string | null | 应收利息 |
| inventories | float | string | null | 存货 |
| amor_exp | float | string | null | 待摊费用 |
| nca_within_1y | float | string | null | 一年内到期的非流动资产 |
| sett_rsrv | float | string | null | 结算备付金 |
| loanto_oth_bank_fi | float | string | null | 拆出资金 |
| premium_receiv | float | string | null | 应收保费 |
| reinsur_receiv | float | string | null | 应收分保账款 |
| reinsur_res_receiv | float | string | null | 应收分保合同准备金 |
| pur_resale_fa | float | string | null | 买入返售金融资产 |
| oth_cur_assets | float | string | null | 其他流动资产 |
| total_cur_assets | float | string | null | 流动资产合计 |
| fa_avail_for_sale | float | string | null | 可供出售金融资产 |
| htm_invest | float | string | null | 持有至到期投资 |
| lt_eqt_invest | float | string | null | 长期股权投资 |
| invest_real_estate | float | string | null | 投资性房地产 |
| time_deposits | float | string | null | 定期存款 |
| oth_assets | float | string | null | 其他资产 |
| lt_rec | float | string | null | 长期应收款 |
| fix_assets | float | string | null | 固定资产 |
| cip | float | string | null | 在建工程 |
| const_materials | float | string | null | 工程物资 |
| fixed_assets_disp | float | string | null | 固定资产清理 |
| produc_bio_assets | float | string | null | 生产性生物资产 |
| oil_and_gas_assets | float | string | null | 油气资产 |
| intan_assets | float | string | null | 无形资产 |
| r_and_d | float | string | null | 研发支出 |
| goodwill | float | string | null | 商誉 |
| lt_amor_exp | float | string | null | 长期待摊费用 |
| defer_tax_assets | float | string | null | 递延所得税资产 |
| decr_in_disbur | float | string | null | 发放贷款及垫款 |
| oth_nca | float | string | null | 其他非流动资产 |
| total_nca | float | string | null | 非流动资产合计 |
| cash_reser_cb | float | string | null | 现金及存放中央银行款项 |
| depos_in_oth_bfi | float | string | null | 存放同业和其它金融机构款项 |
| prec_metals | float | string | null | 贵金属 |
| deriv_assets | float | string | null | 衍生金融资产 |
| rr_reins_une_prem | float | string | null | 应收分保未到期责任准备金 |
| rr_reins_outstd_cla | float | string | null | 应收分保未决赔款准备金 |
| rr_reins_lins_liab | float | string | null | 应收分保寿险责任准备金 |
| rr_reins_lthins_liab | float | string | null | 应收分保长期健康险责任准备金 |
| refund_depos | float | string | null | 存出保证金 |
| ph_pledge_loans | float | string | null | 保户质押贷款 |
| refund_cap_depos | float | string | null | 存出资本保证金 |
| indep_acct_assets | float | string | null | 独立账户资产 |
| client_depos | float | string | null | 其中：客户资金存款 |
| client_prov | float | string | null | 其中：客户备付金 |
| transac_seat_fee | float | string | null | 其中:交易席位费 |
| invest_as_receiv | float | string | null | 应收款项类投资 |
| total_assets | float | string | null | 资产总计 |
| lt_borr | float | string | null | 长期借款 |
| st_borr | float | string | null | 短期借款 |
| cb_borr | float | string | null | 向中央银行借款 |
| depos_ib_deposits | float | string | null | 吸收存款及同业存放 |
| loan_oth_bank | float | string | null | 拆入资金 |
| trading_fl | float | string | null | 交易性金融负债 |
| notes_payable | float | string | null | 应付票据 |
| acct_payable | float | string | null | 应付账款 |
| adv_receipts | float | string | null | 预收款项 |
| sold_for_repur_fa | float | string | null | 卖出回购金融资产款 |
| comm_payable | float | string | null | 应付手续费及佣金 |
| payroll_payable | float | string | null | 应付职工薪酬 |
| taxes_payable | float | string | null | 应交税费 |
| int_payable | float | string | null | 应付利息 |
| div_payable | float | string | null | 应付股利 |
| oth_payable | float | string | null | 其他应付款 |
| acc_exp | float | string | null | 预提费用 |
| deferred_inc | float | string | null | 递延收益 |
| st_bonds_payable | float | string | null | 应付短期债券 |
| payable_to_reinsurer | float | string | null | 应付分保账款 |
| rsrv_insur_cont | float | string | null | 保险合同准备金 |
| acting_trading_sec | float | string | null | 代理买卖证券款 |
| acting_uw_sec | float | string | null | 代理承销证券款 |
| non_cur_liab_due_1y | float | string | null | 一年内到期的非流动负债 |
| oth_cur_liab | float | string | null | 其他流动负债 |
| total_cur_liab | float | string | null | 流动负债合计 |
| bond_payable | float | string | null | 应付债券 |
| lt_payable | float | string | null | 长期应付款 |
| specific_payables | float | string | null | 专项应付款 |
| estimated_liab | float | string | null | 预计负债 |
| defer_tax_liab | float | string | null | 递延所得税负债 |
| defer_inc_non_cur_liab | float | string | null | 递延收益-非流动负债 |
| oth_ncl | float | string | null | 其他非流动负债 |
| total_ncl | float | string | null | 非流动负债合计 |
| depos_oth_bfi | float | string | null | 同业和其它金融机构存放款项 |
| deriv_liab | float | string | null | 衍生金融负债 |
| depos | float | string | null | 吸收存款 |
| agency_bus_liab | float | string | null | 代理业务负债 |
| oth_liab | float | string | null | 其他负债 |
| prem_receiv_adva | float | string | null | 预收保费 |
| depos_received | float | string | null | 存入保证金 |
| ph_invest | float | string | null | 保户储金及投资款 |
| reser_une_prem | float | string | null | 未到期责任准备金 |
| reser_outstd_claims | float | string | null | 未决赔款准备金 |
| reser_lins_liab | float | string | null | 寿险责任准备金 |
| reser_lthins_liab | float | string | null | 长期健康险责任准备金 |
| indept_acc_liab | float | string | null | 独立账户负债 |
| pledge_borr | float | string | null | 其中:质押借款 |
| indem_payable | float | string | null | 应付赔付款 |
| policy_div_payable | float | string | null | 应付保单红利 |
| total_liab | float | string | null | 负债合计 |
| treasury_share | float | string | null | 减:库存股 |
| ordin_risk_reser | float | string | null | 一般风险准备 |
| forex_dinc_min_int | float | string | null | 外币报表折算差额 |
| total_liab_hldr_eqy | float | string | null | 负债及股东权益总计 |
| lt_payroll_payable | float | string | null | 长期应付职工薪酬 |
| oth_comp_income | float | string | null | 其他综合收益 |
| oth_eqt_tools | float | string | null | 其他权益工具 |
| oth_eqt_tools_p_shr | float | string | null | 其他权益工具(优先股) |
| lending_funds | float | string | null | 融出资金 |
| acc_receivable | float | string | null | 应收款项 |
| st_fin_payable | float | string | null | 应付短期融资款 |
| payables | float | string | null | 应付款项 |
| hfs_assets | float | string | null | 持有待售的资产 |
| hfs_sales | float | string | null | 持有待售的负债 |
| cost_fin_assets | float | string | null | 以摊余成本计量的金融资产 |
| fair_value_fin_assets | float | string | null | 以公允价值计量且其变动计入其他综合收益的金融资产 |
| contract_assets | float | string | null | 合同资产 |
| contract_liab | float | string | null | 合同负债 |
| accounts_receiv_bill | float | string | null | 应收票据及应收账款 |
| accounts_pay | float | string | null | 应付票据及应付账款 |
| oth_rcv_total | float | string | null | 其他应收款(合计) |
| fix_assets_total | float | string | null | 固定资产(合计) |
| cip_total | float | string | null | 在建工程(合计) |
| oth_pay_total | float | string | null | 其他应付款(合计) |
| long_pay_total | float | string | null | 长期应付款(合计) |
| debt_invest | float | string | null | 债权投资 |
| oth_debt_invest | float | string | null | 其他债权投资 |
| total_hldr_eqy_exc_min_int | float | string | null | 股东权益合计(不含少数股东权益) |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/balancesheet"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "end_date": "2026-05-11",
    "ann_date": "2026-05-11",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
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
        "total_assets": 9456230000000,
        "total_cur_assets": 3561200000000,
        "total_nca": 5895030000000,
        "total_liab": 8723550000000,
        "total_cur_liab": 5120030000000,
        "total_hldr_eqy_exc_min_int": 724320000000
      }
    ]
  }
}
```
接口名称: 获取现金流量表数据
请求方式: POST
接口地址: /api/stock/cashflow

接口说明:
从 stock_cashflow 表获取现金流量表数据。请求参数 stock_code / end_date / ann_date 三选一至少传一个。每次最多返回 10000 条数据。接口实际返回该表全部字段（除 update_flag、create_time），下方已展示完整字段清单。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| stock_code | string | string[] | 否 | 股票代码，例如 "600000.SH"。支持数组。 |
| end_date | string | 否 | 报告期最后日期，格式 YYYY-MM-DD。 |
| ann_date | string | 否 | 公告日期，格式 YYYY-MM-DD。 |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000，最大10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| stock_code | string | 股票代码（对应 Tushare 的 ts_code） |
| ann_date | string | 公告日期 |
| f_ann_date | string | 实际公告日期 |
| end_date | string | 报告期 |
| comp_type | float | string | null | 公司类型(1一般工商业2银行3保险4证券) |
| report_type | float | string | null | 报表类型 |
| end_type | float | string | null | 报告期类型 |
| net_profit | float | string | null | 净利润 |
| finan_exp | float | string | null | 财务费用 |
| c_fr_sale_sg | float | string | null | 销售商品、提供劳务收到的现金 |
| recp_tax_rends | float | string | null | 收到的税费返还 |
| n_depos_incr_fi | float | string | null | 客户存款和同业存放款项净增加额 |
| n_incr_loans_cb | float | string | null | 向中央银行借款净增加额 |
| n_inc_borr_oth_fi | float | string | null | 向其他金融机构拆入资金净增加额 |
| prem_fr_orig_contr | float | string | null | 收到原保险合同保费取得的现金 |
| n_incr_insured_dep | float | string | null | 保户储金净增加额 |
| n_reinsur_prem | float | string | null | 收到再保业务现金净额 |
| n_incr_disp_tfa | float | string | null | 处置交易性金融资产净增加额 |
| ifc_cash_incr | float | string | null | 收取利息和手续费净增加额 |
| n_incr_disp_faas | float | string | null | 处置可供出售金融资产净增加额 |
| n_incr_loans_oth_bank | float | string | null | 拆入资金净增加额 |
| n_cap_incr_repur | float | string | null | 回购业务资金净增加额 |
| c_fr_oth_operate_a | float | string | null | 收到其他与经营活动有关的现金 |
| c_inf_fr_operate_a | float | string | null | 经营活动现金流入小计 |
| c_paid_goods_s | float | string | null | 购买商品、接受劳务支付的现金 |
| c_paid_to_for_empl | float | string | null | 支付给职工以及为职工支付的现金 |
| c_paid_for_taxes | float | string | null | 支付的各项税费 |
| n_incr_clt_loan_adv | float | string | null | 客户贷款及垫款净增加额 |
| n_incr_dep_cbob | float | string | null | 存放央行和同业款项净增加额 |
| c_pay_claims_orig_inco | float | string | null | 支付原保险合同赔付款项的现金 |
| pay_handling_chrg | float | string | null | 支付手续费的现金 |
| pay_comm_insur_plcy | float | string | null | 支付保单红利的现金 |
| oth_cash_pay_oper_act | float | string | null | 支付其他与经营活动有关的现金 |
| st_cash_out_act | float | string | null | 经营活动现金流出小计 |
| n_cashflow_act | float | string | null | 经营活动产生的现金流量净额 |
| oth_recp_ral_inv_act | float | string | null | 收到其他与投资活动有关的现金 |
| c_disp_withdrwl_invest | float | string | null | 收回投资收到的现金 |
| c_recp_return_invest | float | string | null | 取得投资收益收到的现金 |
| n_recp_disp_fiolta | float | string | null | 处置固定资产、无形资产和其他长期资产收回的现金净额 |
| n_recp_disp_sobu | float | string | null | 处置子公司及其他营业单位收到的现金净额 |
| stot_inflows_inv_act | float | string | null | 投资活动现金流入小计 |
| c_pay_acq_const_fiolta | float | string | null | 购建固定资产、无形资产和其他长期资产支付的现金 |
| c_paid_invest | float | string | null | 投资支付的现金 |
| n_disp_subs_oth_biz | float | string | null | 取得子公司及其他营业单位支付的现金净额 |
| oth_pay_ral_inv_act | float | string | null | 支付其他与投资活动有关的现金 |
| n_incr_pledge_loan | float | string | null | 质押贷款净增加额 |
| stot_out_inv_act | float | string | null | 投资活动现金流出小计 |
| n_cashflow_inv_act | float | string | null | 投资活动产生的现金流量净额 |
| c_recp_borrow | float | string | null | 取得借款收到的现金 |
| proc_issue_bonds | float | string | null | 发行债券收到的现金 |
| oth_cash_recp_ral_fnc_act | float | string | null | 收到其他与筹资活动有关的现金 |
| stot_cash_in_fnc_act | float | string | null | 筹资活动现金流入小计 |
| free_cashflow | float | string | null | 企业自由现金流量 |
| c_prepay_amt_borr | float | string | null | 偿还债务支付的现金 |
| c_pay_dist_dpcp_int_exp | float | string | null | 分配股利、利润或偿付利息支付的现金 |
| incl_dvd_profit_paid_sc_ms | float | string | null | 其中:子公司支付给少数股东的股利、利润 |
| oth_cashpay_ral_fnc_act | float | string | null | 支付其他与筹资活动有关的现金 |
| stot_cashout_fnc_act | float | string | null | 筹资活动现金流出小计 |
| n_cash_flows_fnc_act | float | string | null | 筹资活动产生的现金流量净额 |
| eff_fx_flu_cash | float | string | null | 汇率变动对现金的影响 |
| n_incr_cash_cash_equ | float | string | null | 现金及现金等价物净增加额 |
| c_cash_equ_beg_period | float | string | null | 期初现金及现金等价物余额 |
| c_cash_equ_end_period | float | string | null | 期末现金及现金等价物余额 |
| c_recp_cap_contrib | float | string | null | 吸收投资收到的现金 |
| incl_cash_rec_saims | float | string | null | 其中:子公司吸收少数股东投资收到的现金 |
| uncon_invest_loss | float | string | null | 未确认投资损失 |
| prov_depr_assets | float | string | null | 加:资产减值准备 |
| depr_fa_coga_dpba | float | string | null | 固定资产折旧、油气资产折耗、生产性生物资产折旧 |
| amort_intang_assets | float | string | null | 无形资产摊销 |
| lt_amort_deferred_exp | float | string | null | 长期待摊费用摊销 |
| decr_deferred_exp | float | string | null | 待摊费用减少 |
| incr_acc_exp | float | string | null | 预提费用增加 |
| loss_disp_fiolta | float | string | null | 处置固定、无形资产和其他长期资产的损失 |
| loss_scr_fa | float | string | null | 固定资产报废损失 |
| loss_fv_chg | float | string | null | 公允价值变动损失 |
| invest_loss | float | string | null | 投资损失 |
| decr_def_inc_tax_assets | float | string | null | 递延所得税资产减少 |
| incr_def_inc_tax_liab | float | string | null | 递延所得税负债增加 |
| decr_inventories | float | string | null | 存货的减少 |
| decr_oper_payable | float | string | null | 经营性应收项目的减少 |
| incr_oper_payable | float | string | null | 经营性应付项目的增加 |
| others | float | string | null | 其他 |
| im_net_cashflow_oper_act | float | string | null | 经营活动产生的现金流量净额(间接法) |
| conv_debt_into_cap | float | string | null | 债务转为资本 |
| conv_copbonds_due_within_1y | float | string | null | 一年内到期的可转换公司债券 |
| fa_fnc_leases | float | string | null | 融资租入固定资产 |
| im_n_incr_cash_equ | float | string | null | 现金及现金等价物净增加额(间接法) |
| net_dism_capital_add | float | string | null | 拆出资金净增加额 |
| net_cash_rece_sec | float | string | null | 代理买卖证券收到的现金净额 |
| credit_impa_loss | float | string | null | 信用减值损失 |
| use_right_asset_dep | float | string | null | 使用权资产折旧 |
| oth_loss_asset | float | string | null | 其他资产减值损失 |
| end_bal_cash | float | string | null | 现金的期末余额 |
| beg_bal_cash | float | string | null | 减:现金的期初余额 |
| end_bal_cash_equ | float | string | null | 加:现金等价物的期末余额 |
| beg_bal_cash_equ | float | string | null | 减:现金等价物的期初余额 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/cashflow"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "end_date": "2026-05-11",
    "ann_date": "2026-05-11",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
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
        "n_cashflow_act": 84560000000,
        "n_cashflow_inv_act": -22340000000,
        "n_cash_flows_fnc_act": -41000000000,
        "n_incr_cash_cash_equ": 21220000000,
        "c_cash_equ_end_period": 328900000000
      }
    ]
  }
}
```
接口名称: 获取每日财务数据
请求方式: POST
接口地址: /api/stock/finance

接口说明:
获取指定股票（支持多只）的每日财务指标数据（市盈率、市净率、换手率等）。支持按日期范围筛选。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| stock_code | string | string[] | 否 | 股票代码，例如 "600000.SH"。最大支持100个。 |
| start_time | string | 否 | 开始日期 (YYYY-MM-DD) |
| end_time | string | 否 | 结束日期 (YYYY-MM-DD) |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| stock_code | string | 股票代码 |
| trade_date | string | 交易日期 |
| close | float | 收盘价 |
| turnover_rate | float | 换手率 |
| turnover_rate_f | float | 换手率(自由流通股) |
| volume_ratio | float | 量比 |
| pe | float | 市盈率(总市值/净利润) |
| pe_ttm | float | 市盈率TTM |
| pe_ttm_percentile | float | 市盈率TTM百分位 |
| pb | float | 市净率 |
| ps | float | 市销率 |
| ps_ttm | float | 市销率TTM |
| dv_ratio | float | 股息率 |
| dv_ttm | float | 股息率TTM |
| total_share | float | 总股本 |
| float_share | float | 流通股本 |
| free_share | float | 自由流通股本 |
| total_mv | float | 总市值 |
| circ_mv | float | 流通市值 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/finance"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "start_time": "2023-01-01",
    "end_time": "2026-05-11",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 50,
    "page": 0,
    "page_size": 10000,
    "list": [
      {
        "stock_code": "600000.SH",
        "trade_date": "2023-01-01",
        "close": 10.5,
        "turnover_rate": 0.5,
        "turnover_rate_f": 1.2,
        "volume_ratio": 1.1,
        "pe": 8.5,
        "pe_ttm": 8.2,
        "pe_ttm_percentile": 35.6,
        "pb": 1.1,
        "ps": 2.3,
        "ps_ttm": 2.1,
        "dv_ratio": 3.5,
        "dv_ttm": 3.2,
        "total_share": 1000000,
        "float_share": 800000,
        "free_share": 600000,
        "total_mv": 10500000,
        "circ_mv": 8400000
      }
    ]
  }
}
```
接口名称: 获取股票列表
请求方式: GET
接口地址: /api/stock/list

接口说明:
获取所有股票的基础信息列表（代码、名称、上市日期等）。

请求参数: 无参数

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| stock_code | string | 股票代码 |
| name | string | 股票名称 |
| area | string | 地域 |
| industry | string | 行业 |
| list_date | string | 上市日期 |
| symbol | string | 股票代码（不含后缀） |
| act_name | string | 实控人名称 |
| act_ent_type | string | 实控人企业性质 |
| list_status | string | 上市状态 (L:上市, D:退市, G:过会未交易, P:暂停上市) |
| delist_date | string | 退市日期 |
| is_hs | string | 是否沪深港通标的 (N:否, H:沪股通, S:深股通) |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/list"
headers = {
    "apiKey": "YOUR_API_KEY"
}

response = requests.get(url, headers=headers)
print(response.json())
```

响应示例:
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 5000,
    "list": [
      {
        "stock_code": "000001.SZ",
        "name": "平安银行",
        "area": "深圳",
        "industry": "银行",
        "list_date": "1991-04-03",
        "symbol": "000001",
        "act_name": "深圳市人民政府国有资产监督管理委员会",
        "act_ent_type": "地方国有企业",
        "list_status": "L",
        "delist_date": "",
        "is_hs": "S"
      }
    ]
  }
}
```
接口名称: 获取日K线数据
请求方式: POST
接口地址: /api/stock/daily

接口说明:
获取指定股票（支持多只）的日线级别行情数据（开高低收成交量等）。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| stock_code | string | string[] | 否 | 股票代码，例如 "600000.SH" 或 ["600000.SH", "000001.SZ"]。最大支持100个。如果不传，则返回全市场数据（分页）。 |
| start_time | string | 是 | 开始时间，格式 YYYY-MM-DD |
| end_time | string | 是 | 结束时间，格式 YYYY-MM-DD |
| volType | string | 否 | 成交量单位，可选 "share"(股，默认) 或 "lot"(手) |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| trade_date | string | 交易日期 |
| stock_code | string | 股票代码 |
| stock_name | string | 股票名称 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| pre_close | float | 昨收价 |
| change | float | 涨跌额 |
| pct_chg | float | 涨跌幅(%) |
| vol | float | 成交量（单位由 volType 决定：share=股，lot=手） |
| amount | float | 成交额 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/daily"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "start_time": "2023-01-01",
    "end_time": "2026-05-11",
    "volType": "VALUE",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
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
        "trade_date": "2023-01-01",
        "open": 10.5,
        "high": 11.2,
        "low": 10.4,
        "close": 11,
        "pre_close": 10.4,
        "change": 0.6,
        "pct_chg": 5.77,
        "vol": 50000,
        "amount": 550000,
        "stock_code": "600000.SH"
      }
    ]
  }
}
```
接口名称: 获取周期K线数据
请求方式: POST
接口地址: /api/stock/kline

接口说明:
获取指定股票（支持多只）的周期（周/月）行情数据。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| period | string | 是 | K线周期，可选值: weekly, monthly |
| stock_code | string | string[] | 否 | 股票代码，例如 "600000.SH" 或 ["600000.SH", "000001.SZ"]。 |
| start_time | string | 是 | 开始时间，格式 YYYY-MM-DD |
| end_time | string | 是 | 结束时间，格式 YYYY-MM-DD |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| trade_date | string | 交易日期(通常为该周期最后交易日) |
| stock_code | string | 股票代码 |
| stock_name | string | 股票名称 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| pre_close | float | 昨收价 |
| change | float | 涨跌额 |
| pct_chg | float | 涨跌幅(%) |
| vol | float | 成交量 |
| amount | float | 成交额 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/kline"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "period": "VALUE",
    "stock_code": "600000.SH",
    "start_time": "2023-01-01",
    "end_time": "2026-05-11",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
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
        "trade_date": "2023-01-06",
        "open": 10.5,
        "high": 11.2,
        "low": 10.4,
        "close": 11,
        "pre_close": 10.4,
        "change": 0.6,
        "pct_chg": 5.77,
        "vol": 250000,
        "amount": 2750000,
        "stock_code": "600000.SH"
      }
    ]
  }
}
```
接口名称: 获取周期K线数据(复权)
请求方式: POST
接口地址: /api/stock/kline_adj

接口说明:
获取指定股票的复权周期（周/月）K线数据。目前仅支持前复权(qfq)。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| period | string | 是 | K线周期，可选值: weekly, monthly |
| stock_code | string | 否 | 股票代码，例如 "600000.SH" |
| start_time | string | 否 | 开始时间，格式 YYYY-MM-DD |
| end_time | string | 否 | 结束时间，格式 YYYY-MM-DD |
| algo | string | 否 | 复权算法: "recursive" (默认), "factor" |
| volType | string | 否 | 成交量单位，可选 "share"(股，默认) 或 "lot"(手) |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000，最大10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| stock_code | string | 股票代码 |
| stock_name | string | 股票名称 |
| trade_date | string | 交易日期 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| change | float | 涨跌额 |
| pct_chg | float | 涨跌幅(%) |
| vol | float | 成交量（单位由 volType 决定：share=股，lot=手） |
| amount | float | 成交额 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/kline_adj"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "period": "VALUE",
    "stock_code": "600000.SH",
    "start_time": "2023-01-01",
    "end_time": "2026-05-11",
    "algo": "",
    "volType": "VALUE",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 100,
    "list": [
      {
        "stock_code": "600000.SH",
        "stock_name": "浦发银行",
        "trade_date": "2023-01-06",
        "open": 10.5,
        "high": 11.2,
        "low": 10.4,
        "close": 11,
        "change": 0.6,
        "pct_chg": 5.77,
        "vol": 250000,
        "amount": 2750000
      }
    ]
  }
}
```
接口名称: 获取日K线数据(复权)
请求方式: POST
接口地址: /api/stock/daily_adj

接口说明:
获取指定股票的复权日K线数据。目前仅支持前复权(qfq)。
复权算法说明：
- recursive (递归复权): 默认算法。使用除权除息日的前复权因子进行递归计算。与同花顺算法一致。
- factor (涨跌幅复权): 使用每日的复权因子直接计算。公式：`当前价格 * 当日复权因子`。该算法比较适合量化回测。
注意事项：
- 当股票发生除权除息时，历史的前复权数据需要全部更新。
- 建议使用 `复权因子变更查询` 接口检查当天是否有因子变更，若有，则需要更新对应股票的历史复权数据。
参数说明：
- `stock_code` 和 (`start_time` + `end_time`) 必须至少提供其一。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| stock_code | string | 否 | 股票代码，例如 "600000.SH" |
| start_time | string | 否 | 开始时间，格式 YYYY-MM-DD |
| end_time | string | 否 | 结束时间，格式 YYYY-MM-DD |
| algo | string | 否 | 复权算法: "recursive" (默认), "factor" |
| volType | string | 否 | 成交量单位，可选 "share"(股，默认) 或 "lot"(手) |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000，最大10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| stock_code | string | 股票代码 |
| stock_name | string | 股票名称 |
| trade_date | string | 交易日期 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| change | float | 涨跌额 |
| pct_chg | float | 涨跌幅(%) |
| vol | float | 成交量（单位由 volType 决定：share=股，lot=手） |
| amount | float | 成交额 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/daily_adj"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "start_time": "2023-01-01",
    "end_time": "2026-05-11",
    "algo": "",
    "volType": "VALUE",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 100,
    "list": [
      {
        "stock_code": "600000.SH",
        "trade_date": "2023-01-01",
        "open": 10.35,
        "high": 10.45,
        "low": 10.25,
        "close": 10.4,
        "change": 0.1,
        "pct_chg": 0.97,
        "vol": 50000,
        "amount": 520000
      }
    ]
  }
}
```
接口名称: 获取分钟K线数据(复权)
请求方式: POST
接口地址: /api/stock/min_adj

接口说明:
获取指定股票的分钟级复权K线数据。支持1min/5min原始数据，以及15/30/60分钟聚合数据（从5分钟数据聚合），仅支持前复权(qfq)。
算法说明：
- recursive (递归复权): 默认算法。使用日期对应的复权因子进行递归计算。
- factor (涨跌幅复权): 使用每日的复权因子直接计算。
- 复权因子仅根据日期(YYYY-MM-DD)匹配，不区分具体的分钟时间。
- 15/30/60分钟数据按照A股交易时间对齐（上午09:30-11:30，下午13:00-15:00）。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| stock_code | string | 是 | 股票代码，例如 "600000.SH" |
| level | string | 是 | 数据级别: "1min", "5min", "15min", "30min", "60min" |
| start_time | string | 是 | 开始时间，格式 YYYY-MM-DD HH:MM:SS |
| end_time | string | 是 | 结束时间，格式 YYYY-MM-DD HH:MM:SS |
| algo | string | 否 | 复权算法: "recursive" (默认), "factor" |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| stock_code | string | 股票代码 |
| trade_time | string | 交易时间 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| vol | float | 成交量 |
| amount | float | 成交额 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/min_adj"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "level": "1min",
    "start_time": "2023-01-01",
    "end_time": "2026-05-11",
    "algo": "",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 240,
    "list": [
      {
        "stock_code": "600000.SH",
        "trade_time": "2023-01-01 09:31:00",
        "open": 10.35,
        "high": 10.45,
        "low": 10.25,
        "close": 10.4,
        "vol": 5000,
        "amount": 52000
      }
    ]
  }
}
```
接口名称: 获取复权因子(涨跌幅算法)
请求方式: POST
接口地址: /api/stock/adj_factor

接口说明:
获取指定股票的复权因子数据。复权因子主要用于计算股票的前复权或后复权价格，消除除权除息（分红、配股、拆股等）带来的价格断层影响，保持股价走势的连续性。

计算公式：
- 后复权价格 = 原始价格 × 复权因子
- 前复权价格 = 原始价格 × 复权因子 ÷ 最新复权因子

优点：
1. 真实反映收益：能够真实反映投资者持有股票的实际收益情况。
2. 技术分析准确：消除价格跳空缺口，使均线、MACD等技术指标计算更准确。
3. 策略回测必备：量化交易回测时必须使用复权数据，否则会产生错误的买卖信号。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| stock_code | string | string[] | 否 | 股票代码，例如 "600000.SH"。最大支持100个。 |
| start_time | string | 是 | 开始日期 (YYYY-MM-DD) |
| end_time | string | 是 | 结束日期 (YYYY-MM-DD) |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| stock_code | string | 股票代码 |
| trade_date | string | 交易日期 |
| adj_factor | float | 复权因子 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/adj_factor"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "start_time": "2023-01-01",
    "end_time": "2026-05-11",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 20,
    "page": 0,
    "page_size": 10000,
    "list": [
      {
        "stock_code": "600000.SH",
        "trade_date": "2023-01-01",
        "adj_factor": 12.5678
      }
    ]
  }
}
```
接口名称: 复权因子变更查询
请求方式: POST
接口地址: /api/stock/adj_factor/changes

接口说明:
查询指定日期是否有复权因子变更（即该日期是否为某股票的除权除息日）。
用于判断是否需要更新历史复权数据。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| date | string | 是 | 查询日期 (YYYY-MM-DD) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| data | string[] | 复权因子发生变更的股票代码列表 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/adj_factor/changes"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "date": "2026-05-11"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
```json
{
  "code": 200,
  "msg": "Success",
  "data": [
    "600000.SH",
    "000001.SZ"
  ]
}
```
接口名称: 获取历史分时
请求方式: POST
接口地址: /api/stock/history

接口说明:
获取指定股票的历史分时数据。支持1分钟、5分钟原始数据，以及15/30/60分钟数据。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| stock_code | string | string[] | 是 | 股票代码，例如 "600000.SH"。最大支持100个。 |
| level | string | 是 | 数据级别: "1min", "5min", "15min", "30min", "60min" |
| start_time | string | 是 | 开始时间 (YYYY-MM-DD HH:MM:SS) |
| end_time | string | 是 | 结束时间 (YYYY-MM-DD HH:MM:SS) |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| trade_time | string | 交易时间 |
| stock_code | string | 股票代码 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| vol | float | 成交量(单位手) |
| amount | float | 成交额 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/history"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "level": "1min",
    "start_time": "2023-01-01",
    "end_time": "2026-05-11",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 500,
    "page": 0,
    "page_size": 10000,
    "list": [
      {
        "trade_time": "2023-01-01 09:30:00",
        "open": 10.5,
        "high": 10.6,
        "low": 10.5,
        "close": 10.6,
        "vol": 1000,
        "amount": 10500,
        "stock_code": "600000.SH"
      }
    ]
  }
}
```
接口名称: 下载全市场当天全部分时
请求方式: POST
接口地址: /api/stock/daily_dump

接口说明:
下载全市场当天的全部1分钟或5分钟分时数据，或日线数据，或15/30/60分钟聚合数据。接口返回 GZIP 压缩的 JSON 文件。当 level=daily 时，返回数组格式的日线数据；当 level=1min/5min/15min/30min/60min 时，返回 Map<StockCode, List<Entry>> 格式的分时数据。15/30/60分钟数据从5分钟数据聚合，按照A股交易时间对齐（上午09:30-11:30，下午13:00-15:00）。数据量比较多，一个日期一天最多只能下载10次，超过后会被禁止下载该日期三天，请联系客服解封。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| date | string | 是 | 日期 (YYYY-MM-DD) |
| level | string | 否 | 级别 (daily/1min/5min/15min/30min/60min)。daily返回当天日线数据列表；1min/5min/15min/30min/60min返回分时数据Map，格式为 [时间(HH:MM), 开, 高, 低, 收, 量, 额]。 |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| code | integer | 状态码 (200成功) |
| msg | string | 提示信息 |
| data | array | object | 数据主体。当 level=daily 时，返回数组，每个元素包含完整日线字段；当 level=1min/5min 时，返回对象，Key是股票代码，Value是数组列表，每个数组元素为 [时间(HH:MM), 开, 高, 低, 收, 量, 额]。 |
| data[].trade_date | string | 交易日期 (仅daily级别) |
| data[].stock_code | string | 股票代码 (仅daily级别) |
| data[].open | float | 开盘价 (仅daily级别) |
| data[].high | float | 最高价 (仅daily级别) |
| data[].low | float | 最低价 (仅daily级别) |
| data[].close | float | 收盘价 (仅daily级别) |
| data[].pre_close | float | 昨收价 (仅daily级别) |
| data[].change | float | 涨跌额 (仅daily级别) |
| data[].pct_chg | float | 涨跌幅(%) (仅daily级别) |
| data[].vol | float | 成交量 (仅daily级别) |
| data[].amount | float | 成交额 (仅daily级别) |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/daily_dump"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "date": "2026-05-11",
    "level": "1min"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
```json
{
  "code": 200,
  "msg": "Success",
  "data": [
    {
      "trade_date": "2023-01-01",
      "stock_code": "600000.SH",
      "open": 10.5,
      "high": 10.8,
      "low": 10.4,
      "close": 10.6,
      "pre_close": 10.3,
      "change": 0.3,
      "pct_chg": 2.91,
      "vol": 1000000,
      "amount": 10600000
    },
    {
      "trade_date": "2023-01-01",
      "stock_code": "000001.SZ",
      "open": 12.5,
      "high": 12.8,
      "low": 12.4,
      "close": 12.6,
      "pre_close": 12.3,
      "change": 0.3,
      "pct_chg": 2.44,
      "vol": 2000000,
      "amount": 25200000
    }
  ]
}
```
接口名称: 获取股票停牌信息
请求方式: GET
接口地址: /api/stock/suspension

接口说明:
获取股票停牌信息。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| stock_code | string | 否 | 股票代码 |
| trade_date | string | 否 | 停牌日期 (YYYY-MM-DD) |
| page | integer | 否 | 页码 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| stock_code | string | 股票代码 |
| suspend_date | string | 停牌日期 |
| suspend_start_time | string | 当天停牌开始时间 |
| suspend_end_time | string | 当天停牌结束时间 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/suspension"
headers = {
    "apiKey": "YOUR_API_KEY"
}
params = {
    "stock_code": "600000.SH",
    "trade_date": "2026-05-11",
    "page": 0,
    "page_size": 10000
}

response = requests.get(url, headers=headers, params=params)
print(response.json())
```

响应示例:
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 2,
    "page": 0,
    "page_size": 10000,
    "list": [
      {
        "stock_code": "002462",
        "suspend_date": "2026-01-28",
        "suspend_start_time": null,
        "suspend_end_time": null
      },
      {
        "stock_code": "920159",
        "suspend_date": "2026-01-28",
        "suspend_start_time": "13:03",
        "suspend_end_time": "13:13"
      }
    ]
  }
}
```
接口名称: 获取ST信息
请求方式: POST
接口地址: /api/stock/st_info

接口说明:
获取指定股票的ST信息或按日期获取ST信息列表。股票代码或时间范围必须至少填一个。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| stock_code | string | string[] | 否 | 股票代码，例如 '600069.SH' |
| start_time | string | 否 | 开始日期 (YYYY-MM-DD) |
| end_time | string | 否 | 结束日期 (YYYY-MM-DD) |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| stock_code | string | 股票代码 |
| trade_date | string | 交易日期 |
| stock_name | string | 股票名称 |
| type | string | 类型 (例如 ST) |
| type_name | string | 类型名称 (例如 风险警示板) |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/st_info"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "start_time": "2023-01-01",
    "end_time": "2026-05-11",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
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
接口名称: 获取涨停数据
请求方式: POST
接口地址: /api/stock/limit_up

接口说明:
获取指定股票（支持多只）的涨停明细数据（封单、连板、原因等）。支持按日期范围筛选。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| stock_code | string | string[] | 否 | 股票代码，例如 '600000.SH'。 |
| start_time | string | 是 | 开始日期 (YYYY-MM-DD) |
| end_time | string | 是 | 结束日期 (YYYY-MM-DD) |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000，最大10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| trade_date | string | 交易日期 |
| stock_code | string | 股票代码 |
| stock_name | string | 股票名称 |
| price | float | 最新价/收盘价 |
| change_percent | float | 涨跌幅(%) |
| first_limit_time | string | 首次涨停时间 |
| final_limit_time | string | 最终涨停时间 |
| consecutive_days | integer | 连续涨停天数 |
| sealed_volume | float | 封单量 |
| sealed_amount | float | 封单额 |
| sealed_turnover_ratio | float | 封成比 |
| sealed_flow_ratio | float | 封流比 |
| open_count | integer | 开板次数 |
| boards | integer | 几天几板中的板数 |
| limit_type | string | 涨停类型 |
| is_limit_up | integer | 是否涨停(1是0否) |
| reason_text | string | 涨停原因 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/limit_up"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "start_time": "2023-01-01",
    "end_time": "2026-05-11",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
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
        "sealed_amount": 22152780,
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
接口名称: 获取同花顺热度榜
请求方式: GET / POST
接口地址: /api/ths/hot

接口说明:
获取同花顺热度榜数据。支持传 market 指定榜单类型，支持传 trade_date 指定交易日；不传 trade_date 默认返回该市场最新交易日数据。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| market | string | 否 | 热榜类型 (默认：热股)。可选值：热股, ETF, 可转债, 行业板块, 概念板块, 期货 |
| trade_date | string | 否 | 指定交易日期，支持 YYYY-MM-DD 或 YYYYMMDD。GET 用 Query，POST 用 JSON Body |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| trade_date | string | 交易日期 |
| update_time | string | 排行榜更新时间 |
| list | array | 热榜数据列表 |
| list[].name | string | 名称 |
| list[].code | string | 代码 |
| list[].rank | integer | 排名 |
| list[].pct_change | float | 涨跌幅(%) |
| list[].hot | float | 热度值 |

请求示例 (Python):
```python
# 暂无示例
```

响应示例:
```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "trade_date": "20260416",
    "update_time": "2026-04-16 22:30:00",
    "list": [
      {
        "name": "浦发银行",
        "code": "600000.SH",
        "rank": 1,
        "pct_change": 2.5,
        "hot": 98.5
      }
    ]
  }
}
```
接口名称: 获取涨跌停数据
请求方式: POST
接口地址: /api/stock/limit_list

接口说明:
获取指定股票（支持多只）的涨跌停数据（封板时间、封单金额、连板状态等）。支持按日期范围筛选。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| stock_code | string | string[] | 否 | 股票代码，例如 '600000.SH'。最大支持100个。 |
| start_time | string | 是 | 开始日期 (YYYY-MM-DD) |
| end_time | string | 是 | 结束日期 (YYYY-MM-DD) |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量 (默认10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| trade_date | string | 交易日期 |
| stock_code | string | 股票代码 |
| name | string | 股票名称 |
| industry | string | 所属行业 |
| close | float | 收盘价 |
| pct_chg | float | 涨跌幅(%) |
| amount | float | 成交额 |
| limit_amount | float | 板上成交额 |
| float_mv | float | 流通市值 |
| total_mv | float | 总市值 |
| turnover_ratio | float | 换手率 |
| fd_amount | float | 封单金额 |
| first_time | string | 首次封板时间 |
| last_time | string | 最后封板时间 |
| open_times | integer | 打开次数 |
| up_stat | string | 连板状态 |
| limit_times | float | 连板数 |
| limit | string | 涨跌停状态(U涨停 D跌停 Z炸板) |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/limit_list"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "start_time": "2023-01-01",
    "end_time": "2026-05-11",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
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
        "pct_chg": 10,
        "amount": 4089780784,
        "limit_amount": null,
        "float_mv": 69517045970.24,
        "total_mv": 69533895735.04,
        "turnover_ratio": 6.07,
        "fd_amount": 31512052,
        "first_time": "142900",
        "last_time": "145503",
        "open_times": 1,
        "up_stat": "1/1",
        "limit_times": 1,
        "limit": "U"
      }
    ]
  }
}
```
接口名称: 获取大小单资金金流向
请求方式: POST
接口地址: /api/stock/main_fund_flow

接口说明:
获取大小单资金金流向数据。分档口径：小单<5万，中单5万~20万，大单20万~100万，特大单>=100万。查询条件支持仅传(stock_code)或仅传(start_time+end_time)或两者同时传；时间区间为闭区间，start_time=end_time时可查询当天数据。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| start_time | string | 否 | 开始日期，格式 YYYY-MM-DD（与end_time配套，可只传时间范围） |
| end_time | string | 否 | 结束日期，格式 YYYY-MM-DD（与start_time配套；闭区间，start_time=end_time可查当天） |
| stock_code | string | string[] | 否 | 股票代码，例如 "600000.SH" 或 "600000" 或 ["600000.SH", "000001.SZ"] |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量，最大10000 (默认10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| trade_date | string | 交易日期 |
| stock_code | string | 股票代码 |
| buy_sm_vol | float | 小单买入量（手） |
| buy_sm_amount | float | 小单买入金额（万元） |
| sell_sm_vol | float | 小单卖出量（手） |
| sell_sm_amount | float | 小单卖出金额（万元） |
| buy_md_vol | float | 中单买入量（手） |
| buy_md_amount | float | 中单买入金额（万元） |
| sell_md_vol | float | 中单卖出量（手） |
| sell_md_amount | float | 中单卖出金额（万元） |
| buy_lg_vol | float | 大单买入量（手） |
| buy_lg_amount | float | 大单买入金额（万元） |
| sell_lg_vol | float | 大单卖出量（手） |
| sell_lg_amount | float | 大单卖出金额（万元） |
| buy_elg_vol | float | 特大单买入量（手） |
| buy_elg_amount | float | 特大单买入金额（万元） |
| sell_elg_vol | float | 特大单卖出量（手） |
| sell_elg_amount | float | 特大单卖出金额（万元） |
| net_mf_vol | float | 净流入量（手） |
| net_mf_amount | float | 净流入额（万元） |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/main_fund_flow"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "start_time": "2023-01-01",
    "end_time": "2026-05-11",
    "stock_code": "600000.SH",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
```json
{
  "code": 200,
  "msg": "成功",
  "data": {
    "total": 1,
    "list": [
      {
        "trade_date": "2026-04-03",
        "stock_code": "600000.SH",
        "buy_sm_vol": 12456,
        "buy_sm_amount": 245.88,
        "sell_sm_vol": 10234,
        "sell_sm_amount": 201.43,
        "buy_md_vol": 8543,
        "buy_md_amount": 167.56,
        "sell_md_vol": 7321,
        "sell_md_amount": 152.12,
        "buy_lg_vol": 4321,
        "buy_lg_amount": 98.71,
        "sell_lg_vol": 3210,
        "sell_lg_amount": 77.83,
        "buy_elg_vol": 2109,
        "buy_elg_amount": 64.23,
        "sell_elg_vol": 1980,
        "sell_elg_amount": 51.19,
        "net_mf_vol": 5684,
        "net_mf_amount": 93.81
      }
    ]
  }
}
```
接口名称: 获取筹码峰分布
请求方式: POST
接口地址: /api/stock/cyq_chips

接口说明:
获取筹码峰分布数据。可按时间范围或股票代码查询，支持只传其一或同时传。时间区间为闭区间，start_time=end_time 时可查询当天数据。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| start_time | string | 否 | 开始日期，格式 YYYY-MM-DD（与end_time配套） |
| end_time | string | 否 | 结束日期，格式 YYYY-MM-DD（与start_time配套；闭区间） |
| stock_code | string | string[] | 否 | 股票代码，例如 "600000.SH" 或 "600000" 或 ["600000.SH", "000001.SZ"] |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量，最大10000 (默认10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| trade_date | string | 交易日期 |
| stock_code | string | 股票代码 |
| price | float | 成本价格 |
| percent | float | 价格占比 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/cyq_chips"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "start_time": "2023-01-01",
    "end_time": "2026-05-11",
    "stock_code": "600000.SH",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
```json
{
  "code": 200,
  "msg": "成功",
  "data": {
    "total": 2,
    "list": [
      {
        "trade_date": "2026-04-03",
        "stock_code": "600000.SH",
        "price": 9.45,
        "percent": 0.012345
      },
      {
        "trade_date": "2026-04-03",
        "stock_code": "600000.SH",
        "price": 9.46,
        "percent": 0.018765
      }
    ]
  }
}
```
接口名称: 龙虎榜机构明细
请求方式: POST
接口地址: /api/stock/dragon_tiger

接口说明:
查询龙虎榜机构明细数据。date 和 stock_code 必须至少提供一个。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| date | string | 否 | 交易日期 (YYYY-MM-DD) |
| stock_code | string | 否 | 股票代码 (e.g. 600519.SH) |
| page | integer | 否 | 页码，从1开始 (默认1) |
| page_size | integer | 否 | 每页数量，最大1000 (默认20) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| trade_date | string | 交易日期 |
| stock_code | string | 股票代码 |
| org_name | string | 营业部名称 |
| buy_amount | float | 买入额(万) |
| buy_ratio | float | 买入占比 |
| sell_amount | float | 卖出额(万) |
| sell_ratio | float | 卖出占比 |
| net_buy_amount | float | 净买入额(万) |
| direction | integer | 买卖方向 (0:未知, 1:买入, 2:卖出) |
| reason | string | 上榜理由 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/dragon_tiger"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "date": "2026-05-11",
    "stock_code": "600000.SH",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
```json
{
  "code": 200,
  "msg": "Success",
  "data": [
    {
      "trade_date": "2026-03-16",
      "stock_code": "000533.SZ",
      "org_name": "东方财富证券股份有限公司拉萨团结路第一证券营业部",
      "buy_amount": 1868.9,
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
接口名称: 龙虎榜每日明细
请求方式: POST
接口地址: /api/stock/top_list

接口说明:
查询龙虎榜每日交易明细数据。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| trade_date | string | 是 | 交易日期 (YYYY-MM-DD) |
| stock_code | string | 否 | 股票代码 (e.g. 600519.SH) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| trade_date | string | 交易日期 |
| stock_code | string | 股票代码 |
| name | string | 名称 |
| close | float | 收盘价 |
| pct_change | float | 涨跌幅 |
| turnover_rate | float | 换手率 |
| amount | float | 总成交额 |
| l_sell | float | 龙虎榜卖出额 |
| l_buy | float | 龙虎榜买入额 |
| l_amount | float | 龙虎榜成交额 |
| net_amount | float | 龙虎榜净买入额 |
| net_rate | float | 龙虎榜净买额占比 |
| amount_rate | float | 龙虎榜成交额占比 |
| float_values | float | 当日流通市值 |
| reason | string | 上榜理由 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/top_list"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "trade_date": "2026-05-11",
    "stock_code": "600000.SH"
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
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
接口名称: 获取券商盈利预测
请求方式: POST
接口地址: /api/stock/report_rc

接口说明:
获取券商（卖方）每天研报的盈利预测数据。stock_code / report_date / (start_date+end_date) 三类条件至少传一个；如果传时间区间，start_date 与 end_date 必须同时传。

请求参数:
| 参数名 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| stock_code | string | 否 | 股票代码 |
| report_date | string | 否 | 报告日期 (YYYYMMDD 或 YYYY-MM-DD) |
| start_date | string | 否 | 报告开始日期 (YYYYMMDD 或 YYYY-MM-DD)，与 end_date 配套 |
| end_date | string | 否 | 报告结束日期 (YYYYMMDD 或 YYYY-MM-DD)，与 start_date 配套 |
| page | integer | 否 | 页码，从0开始 (默认0) |
| page_size | integer | 否 | 每页数量，最大10000 (默认10000) |

响应参数:
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| stock_code | string | 股票代码 |
| name | string | 股票名称 |
| report_date | string | 研报日期 |
| report_title | string | 报告标题 |
| report_type | string | 报告类型 |
| classify | string | 报告分类 |
| org_name | string | 机构名称 |
| author_name | string | 作者 |
| quarter | string | 预测报告期 |
| op_rt | float | 预测营业收入（万元） |
| op_pr | float | 预测营业利润（万元） |
| tp | float | 预测利润总额（万元） |
| np | float | 预测净利润（万元） |
| eps | float | 预测每股收益（元） |
| pe | float | 预测市盈率 |
| rd | float | 预测股息率 |
| roe | float | 预测净资产收益率 |
| ev_ebitda | float | 预测EV/EBITDA |
| rating | string | 卖方评级 |
| max_price | float | 预测最高目标价 |
| min_price | float | 预测最低目标价 |
| imp_dg | string | 机构关注度 |

请求示例 (Python):
```python
import requests

url = "https://data.diemeng.chat/api/stock/report_rc"
headers = {
    "apiKey": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
payload = {
    "stock_code": "600000.SH",
    "report_date": "2026-05-11",
    "start_date": "2023-01-01",
    "end_date": "2026-05-11",
    "page": 0,
    "page_size": 10000
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

响应示例:
```json
{
  "code": 200,
  "msg": "Success",
  "data": {
    "total": 1,
    "page": 0,
    "page_size": 10000,
    "list": [
      {
        "stock_code": "600066.SH",
        "name": "宇通客车",
        "report_date": "2025-01-08",
        "report_title": "宇通客车：12月高景气度延续，2024全年产销超预期",
        "report_type": "点评",
        "classify": "一般报告",
        "org_name": "东吴证券",
        "author_name": "黄细里,孙仁昊",
        "quarter": "2024Q4",
        "op_rt": 3473300,
        "op_pr": null,
        "tp": 423600,
        "np": 355381,
        "eps": 1.61,
        "pe": 17.21,
        "rd": null,
        "roe": 27.24,
        "ev_ebitda": null,
        "rating": "买入",
        "max_price": null,
        "min_price": null,
        "imp_dg": ""
      }
    ]
  }
}
```
