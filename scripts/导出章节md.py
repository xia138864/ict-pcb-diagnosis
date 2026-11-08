# -*- coding: utf-8 -*-
"""从 论文_修订版.docx 导出章节 md（保证与 Word 完全一致）"""
import docx
import io

OUT = 'C:/Users/Administrator/WorkBuddy/2026-09-07-23-02-54/'
d = docx.Document(OUT + '论文_修订版.docx')

# 由于 python-docx 中表格与段落是分离的，这里按文档体顺序遍历
from docx.oxml.ns import qn
body = d.element.body
para_map = {p._element: p for p in d.paragraphs}
tbl_map = {t._element: t for t in d.tables}

items = []   # (kind, text)
for child in body.iterchildren():
    if child.tag == qn('w:p'):
        p = para_map.get(child)
        if p is not None and p.text.strip():
            items.append(('p', p.text.strip()))
    elif child.tag == qn('w:tbl'):
        t = tbl_map.get(child)
        if t is not None:
            items.append(('t', t))

# 找章节边界
def idx_of(pred, start=0):
    for i in range(start, len(items)):
        if items[i][0] == 'p' and pred(items[i][1]):
            return i
    return -1

i_title = 0
i_abs = idx_of(lambda s: s == '摘要')
i_1 = idx_of(lambda s: s.startswith('1 引言'))
i_2 = idx_of(lambda s: s.startswith('2 相关研究'))
i_3 = idx_of(lambda s: s.startswith('3 方法'))
i_4 = idx_of(lambda s: s.startswith('4 实验设置'))
i_6 = idx_of(lambda s: s.startswith('6 讨论'))
i_ref = idx_of(lambda s: s == '参考文献')


def render(lo, hi):
    out = []
    ti = 0
    for kind, v in items[lo:hi]:
        if kind == 'p':
            out.append(v)
            out.append('')
        else:
            # 表格转 md
            rows = [[c.text.strip() for c in r.cells] for r in v.rows]
            out.append('| ' + ' | '.join(rows[0]) + ' |')
            out.append('|' + '|'.join(['---'] * len(rows[0])) + '|')
            for r in rows[1:]:
                out.append('| ' + ' | '.join(r) + ' |')
            out.append('')
            ti += 1
    return '\n'.join(out)


# 1) 摘要 + 引言
md1 = ['# 摘要与引言（与 论文_修订版.docx 同步导出）', '',
       '> ⚠️ 本文件由 Word 自动导出，保证与正文完全一致。修改请改 `生成论文word.py` 后重新生成。', '']
md1.append(render(i_abs, i_2))
io.open(OUT + '论文章节_摘要引言.md', 'w', encoding='utf-8').write('\n'.join(md1))

# 2) 方法与实验（第 3~5 章）
md2 = ['# 方法与实验章节（与 论文_修订版.docx 同步导出）', '',
       '> ⚠️ 本文件由 Word 自动导出，保证与正文完全一致。修改请改 `生成论文word.py` 后重新生成。', '']
md2.append(render(i_3, i_6))
io.open(OUT + '论文章节_方法与实验.md', 'w', encoding='utf-8').write('\n'.join(md2))

print('已导出: 论文章节_摘要引言.md / 论文章节_方法与实验.md')
print('段落项:', len(items), '| 摘要区间', i_abs, i_2, '| 方法区间', i_3, i_6)
