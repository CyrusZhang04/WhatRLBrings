# -*- coding: utf-8 -*-
# Original analysis script, kept for reference (see legacy/README.md).
# Internal paths were replaced by ${WRB_*} placeholders; this file reads
# intermediate artifacts that are not in the repository.
import re
def body(f):
    t=open(f).read(); return t.split('---\n\n',1)[1].rstrip()
def cut(t,a,b,name):
    i=t.find(a); assert i>=0, f'{name}: 找不到起点 {a[:30]!r}'
    j=t.find(b,i); assert j>=0, f'{name}: 找不到终点 {b[:30]!r}'
    return t[i:j+len(b)].strip()
A=body('annot_aime24_q15_s6.md'); B=body('annot_aime25_q4_s8.md')
out=[]
out.append("""# 图例摘录（2026-10-14 导出）

粗体 = RL 换入的 token（`base_argmax != rl_argmax` 的位置，内容取 RL 的 argmax），其余 token 由 base 自己采样。
原文与逐处替换见同目录 annot_*.md / reps_*.tsv。

## A. 双曲线菱形题 —— AIME24 #15（答案 480）

口径提醒：**这题 base 0/16、rl-replace 0/16、RL 0/16，没有任何一组做对**，只能当"过程与风格"的例子。
base 最常给 80（8/16），rl-replace 给 80（4）、960（3）、959（2），RL 给 80（5）、959（3）、960（2）。
下面取 rl-replace 样本 6：1611 token，替换 76 处 = 4.72%，token 级对齐 76/76。
""")
out.append('### A1 开头\n\n'+cut(A,'**Alright**, I have this problem','understand **it first**.','A1'))
out.append('### A2 列出菱形性质\n\n'+cut(A,'**Breaking It Down:**','Diagonals intersect at the origin.','A2'))
out.append('### A3 一处 Wait 自检（base 原本要写 `So`，RL 换成 `Wait`）\n\n'
           +cut(A,'Since the diagonals **of** the rhombus','**positions** of **\\(** A \\), \\( B \\), \\( C \\), and \\( D \\).','A3')
           +'\n\n注：这处自检提出的质疑是对的（不能默认两条对角线就在坐标轴上），但回答随后仍沿用了坐标轴假设，所以最终答成 959。')
out.append('### A4 结尾\n\n'+cut(A,'We have **found** that \\( BD^2 = 960 \\)','\\boxed{959}\n\\]','A4'))
out.append("""### A5 base 回答的开头几行（同题，供并排）

base 样本 0（1083 token，错，最终答 `\\boxed{\\text{Unbounded}}`）：

> We are given the hyperbola \\(\\frac{x^2}{20} - \\frac{y^2}{24} = 1\\) and four points \\(A\\), \\(B\\), \\(C\\), and \\(D\\) on this hyperbola such that \\(ABCD\\) forms a rhombus with diagonals intersecting at the origin. We need to find the greatest real number that is less than \\(BD^2\\) for all such rhombi.
>
> ### Step 1: Understanding the Hyperbola and Rhombus

base 样本 1（2215 token，错）：

> Assistant: To solve the problem, we'll follow these steps:
>
> 1. **Understand the Hyperbola and Rhombus:** …
> 2. **Properties of the Rhombus:** …

对比点：base 以"We are given / To solve the problem"开场、直接列步骤；rl-replace 第一个 token 就被换成 `Alright`（base 原本要写 `Assistant` 或 `We`），之后是第一人称叙述 + 自我质疑。

## B. Wait 真正纠错的题 —— AIME25 #4（答案 117）

base 0/16、rl-replace 10/16、RL 15/16。base 的答案分布 168（6）、152（2）、118（2）、46（2）、50（1）、34（1），**一条 117 都没有**；rl-replace 117（10）、118（2）、101（2）、129（1）、267（1）；RL 117（15）、101（1）。
下面取 rl-replace 样本 8：2085 token，替换 84 处 = 4.03%，对齐 84/84，**全文只插入了一处 `Wait`，就是纠错那处**。
""")
out.append('### B1 开头\n\n'+cut(B,'**Alright**, I have this problem','Let\'s break **it** down**.**','B1'))
out.append('### B2 两种情形各自计数，得到 118\n\n'+cut(B,'### Total Number of Ordered Pairs','\\[ 51 + 67 = 118 \\]','B2'))
out.append('### B3 Wait 纠错（base 原本要写 `###`，即直接开新小节收尾 118；RL 换成 `Wait`）\n\n'
           +cut(B,'**Wait**, but I need to check','\\[ 51 + 67 - 1 = 117 \\]','B3'))
out.append('### B4 结尾\n\n'+cut(B,'### Final Answer','\\boxed{117}\\]','B4'))
out.append("""### B5 base 回答的开头几行（同题，供并排）

base 样本 0（1070 token，错，最终答 50）：

> To solve the problem of finding the number of ordered pairs \\((x, y)\\) where both \\(x\\) and \\(y\\) are integers between \\(-100\\) and \\(100\\) inclusive, such that the equation \\(12x^2 - xy - 6y^2 = 0\\) holds, we will follow these steps:
>
> 1. **Factorize the quadratic equation:** …

base 样本 2（923 token，错）：

> ### Step 1: Factor the Equation … \\(12x^2 - xy - 6y^2 = (3x + 2y)(4x - 3y) = 0\\)

对比点：base 这题也能正确分解出两族解，但从不回头查两族的重叠，答案落在 168 / 152 / 50 这类数上；rl-replace 在"写完 118、准备开新小节"的那一个 token 上被换成 `Wait`，接着查重叠、减掉 (0,0)，得到 117。
""")
open('excerpts_20261014.md','w').write('\n\n'.join(out)+'\n')
print('\n\n'.join(out))
