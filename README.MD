# NAGA/Mortal牌谱解析

这是[NAGA助手](https://ricochet.cn/riichi/naga)网站NAGA/Mortal牌谱解析的代码。

之所以开源这部分代码，是因为其技术含金量最高（有实际价值）、计算量较重（缓解网站负载）、且欢迎各位贡献。网站其余部分主要是重CRUD逻辑，参考价值不大，因此不做开源。

## 使用方式
```shell
python naga_session.py <naga牌谱网址> <Mortal牌谱id>
```
Mortal牌谱id为选填。

格式参考：
```shell
python naga_session.py https://naga.dmv.nico/htmls/9a852e2e3a273e1d5a362d69618000684340d3e0e96024e6fa07858afa1afa00v2_2.html 01f7faef6dc7fe5e

```

## 功能
1. NAGA牌谱报告分析
2. Mortal牌谱报告分析 & 把Mortal报告融合入NAGA界面
3. NAGA网页端快捷键注入脚本
