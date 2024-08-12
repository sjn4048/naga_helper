# NAGA/Mortal牌谱解析

这是 [NAGA助手](https://ricochet.cn/riichi/naga) 网站NAGA/Mortal牌谱解析的代码。

之所以开源这部分代码，是因为其技术含金量最高（有实际价值）、计算量较重（缓解网站负载）、且欢迎各位贡献。网站其余部分主要是重CRUD逻辑，参考价值不大，因此不做开源。

## Install
```shell
pip install naga_helper
```

## 模块
### analyzer
分析现有NAGA牌谱报告，使用方式如下：

导入pip包：
```python
from naga_helper.analyzer import parse_report
parse_report('NAGA html content')
```

命令行：
```shell
python naga_helper/analyzer.py <naga牌谱网址> <Mortal牌谱id>
```
其中Mortal牌谱id为选填，如果填写，则会把Mortal的切牌选择合并入NAGA牌谱报告。恶手率/一致率/Rating的计算方式与NAGA一致。

naga牌谱地址可以同时省略前缀与后缀，只保留html key。

格式参考：
```shell
python naga_helper/analyzer.py https://naga.dmv.nico/htmls/9a852e2e3a273e1d5a362d69618000684340d3e0e96024e6fa07858afa1afa00v2_2.html 01f7faef6dc7fe5e

```

### plugins
各类插件，包括NAGA网页模块等

## 功能
1. NAGA牌谱报告分析
2. Mortal牌谱报告分析 & 把Mortal报告融合入NAGA界面
3. NAGA网页端快捷键注入脚本

