from glob import glob

posts = glob('./_posts/*.md')
link_template = '[%s]: %s site.baseurl %s link %s %s'
title_template = '[{}][{}]'

links = []
titles = []
for post in posts:
    post_name = post.split('-')[-1][:-3]
    with open(post, encoding='utf-8') as f:
        next(f)
        post_title = next(f).split('title:')[1].strip()

    link = link_template % (post_name, '{{', '}}{%', post[2:], '%}')
    links.append(link)

    title = title_template.format(post_title, post_name)
    titles.append(title)


with open('page_summary.txt', 'w', encoding='utf-8') as f:
    for title in titles:
        f.write('{}\n'.format(title))
    f.write('\n\n')
    for link in links:
        f.write('{}\n'.format(link))