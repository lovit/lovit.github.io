---
title: Github 으로 텍스트 문서 버전 관리하기
date: 2018-08-17 20:00:00
categories:
- analytics
tags:
- analytics
---

이전에 글을 쓸 때 첨삭의 편의 때문에 Word 를 이용했습니다. 그런데 여러 명이 함께 글을 작업하다보니 버전이 꼬이는 문제가 발생했습니다. 그 때 부터 문서에 대한 버전 관리의 필요를 느끼게 되었고, 최근에 github 을 이용하여 문서 버전 관리를 하기 시작했습니다. 그 과정에서 배운 점들을 정리합니다. 조금 더 효율적인 문서의 버전 관리를 할 수 있기를 바라며 글을 공유합니다.

## 사건의 발단, Word 에서 github 으로

동료와 함께 글 하나를 쓰는 중이었습니다. 이전에는 글을 쓸 때 첨삭이 편하다는 이유로 Word 를 이용했습니다. 이제와 생각해 보니 물론 Microsoft Word 의 comment 기능이 편리하기는 합니다만, 유일하게 그 기능을 제공하는 툴도 아닙니다. 쓰던 것이 편하다는 일종의 레거시라고 생각됩니다. 다른 툴을 고민하지 않고서 여전히 Word 를 쓰던 것이었으니까요.

여하튼 Word 를 이용하여 문서 버전을 관리하다보니 아래처럼 파일로 버전 관리를 하게 되었습니다. 사실 이 스크린샷에서도 알 수 있는 단점은, 각 버전마다 무엇이 크게 바뀌었는지 문서를 열어보기 전에는 알 수가 없다는 점입니다. 적어도 파일명에 큰 변화는 알 수 있게 메모를 하거나, 버전 별 readme 라도 만들어 뒀어야 했습니다. 일단 이 점이 잘못된 점이었습니다.

![]({{ "/assets/figures/latex_with_github_list_of_vers.png" | absolute_url }}){: width="40%" height="40%"}

물론 워드로도 충분히 첨삭되는 부분을 잘 추적하며 관리할 수 있지만, 문제는 두 명이 동시에 문서를 건드리기 시작하면서 발생했습니다. 사본 충돌의 가능성이 있기 때문에 파일을 두 개로 복사하여 나눠 작업을 시작하였습니다. 완료가 되는대로 한 명이 수정된 내용을 반영하기로 했습니다. 그리고는 수정된 내용을 반영해야 했던 저는 그 일을 잊어버렸습니다. 한 부분의 수정이 반영되지 않은 체 일은 계속 진행되었습니다.

다른 문제도 생겼습니다. 동 버전의 여러 파일에 나눠서 서로가 조금씩 내용을 바꿨습니다. Word 파일로 작업을 할 때에는 수정할 범위를 명시하고 그 부분에 대해서만 수정해야 하는데, 그런 생각도 하지 않았습니다. 결국 최종본에는 애써 고쳤던 내용들이 군대군대 빠져있는 일이 생겼습니다. 앞서 말한 것처럼 일이라도 몰아서 한 번에 끝내는 성격이면 무엇인가 놓쳤다는 사실을 빨리 깨달았을텐데 그것도 아니어서, 엄두가 안났습니다. 문서 두 개 펼쳐두고 틀린 부분 찾자니 그것도 싫었고, 그걸 위해 두 문서의 틀린 부분을 트래킹 할 수 있는 코드를 짜자니 자괴감이 들었습니다.

"일을 몰아서 했었더라면" 이란 생각도 잘못된 생각입니다. 애초에 이런 문제가 발생하지 않도록 잘 준비를 했었어야 했습니다. 다른 이슈도 있었습니다. 예전에 논문 첨삭 받았던 첨삭본을 잃어버린 적도 있었습니다. 종이에 팬으로 기록한 정보는 물리적으로 잃어버리기 쉽습니다.

그 때 부터 어떤 방법이 좋을까 고민했습니다. 코드도 버전을 관리하는데, 문서 역시 버전을 관리 해야겠다는 생각을 했고, 그런 툴을 찾아봤습니다.

이전에 들었던 이야기가 생각났습니다. 어떤 연구실의 교수님은 학생들의 논문 리뷰를 github 으로 해주신다는 이야기를 들었었는데, 생각해보니 이점이 많아 보였습니다. github 은 difference 에 대해서 tracking 을 할 수 있고, 수정된 부분마다 commit 을 하면 어떤 부분이 수정된 것인지, 그리고 그 때의 버전으로 문서를 볼 수도 있으니까요. 저도 git 으로 문서 버전을 관리하는 연습을 해보기로 했습니다.

기대하는 효과는 1) 문서의 수정 사항들을 관리할 수 있고, 2) 각 버전마다 문서를 복원할 수 있으며, 3) 인쇄본을 잃어버린다던지 실수로 폴더를 지워 문서를 잃어버리는 일을 방지할 수 있다는 점입니다.

## Github 으로 문서 버전을 관리하기 위한 팁들

일단, 수정을 해야 하는 latex 파일이 하나 있었습니다. 이 파일을 수정하면서 git 으로 latex 를 관리하는 연습을 해보기로 했습니다.

### 수정의 내용 (주제) 단위로 commit 하기

원칙은 코드 관리와 동일합니다. 수정하는 내용 단위로 commit 을 넣습니다. 그 내용이 문서의 중간과 끝부분에 나뉘어져 있다 하더라도 내용 단위로 commit 을 남기면, 무엇을 수정했는지 이후에 묶어서 볼 수가 있습니다.

![]({{ "/assets/figures/latex_with_github_commit.png" | absolute_url }})

아래 그림은 예시 latex 코드에서 가독을 위하여 section 마다 넣어두었던 주석줄의 길이를 일괄적으로 줄이는 부분입니다. 문서의 위치 단위가 아니라 의미 단위로 수정을 하면 이후에 관리도 쉽습니다.

![]({{ "/assets/figures/latex_with_github_diff.png" | absolute_url }})

또한 commit 을 누르면 해당 commit 에 대한 comments 를 달 수 있습니다. 메모를 남기고, 함께 작업하는 사람과 간단한 토론을 할 수도 있습니다.

### Branch 로 큰 주제 관리하기

큰 글의 경우에는, 혹은 여러 개의 단편적인 글들을 한 번에 작업할 때에는 branch 를 나눠 관리할 수도 있습니다. Branch 를 나누면 큰 주제에 대한 commit 만을 보며 관리할 수 있으니까요. 수정된 최종 버전은 master 에 merge 하면 됩니다.

![]({{ "/assets/figures/latex_with_github_branch.png" | absolute_url }})

### Issue board 를 이용하여 todo list 관리하기

이후로 해야 할 일들은 issue board 를 이용하기로 했습니다. 여러 명이서 작업하는 문서라면 누군가에게 할 일을 할당할 수도 있습니다. 일이 완료되면 해당 issue 를 closed 하면 됩니다.

포스트잇에 todo list 를 적고 줄을 긋는 것 보다 훨씬 현명한 방법이라 생각됩니다. Issue boards 의 숫자를 보며 스스로 압박도 받는다는 장점(?)도 있고요.

![]({{ "/assets/figures/latex_with_github_issueboard.png" | absolute_url }})

### 한 문장을 한 줄로 적기

Latex 을 쓸 때 한 단락을 한 줄로 적는 경우가 많습니다. 그런데 git 은 줄 단위로 코드를 관리합니다. 열 문장으로 이뤄진 한 단락을 한 줄에 모두 적으면 한 줄만 바꿔도 단락 전체가 바뀐 것으로 diff 에 표시됩니다.

Latex 은 빈 줄을 기준으로 단락을 나눕니다. 붙어있는 연속된 두 줄은 하나의 단락으로 인식합니다. 그러므로 한 문장을 한 줄로 적는다면, 열 문장으로 이뤄진 단락에서 한 문장을 수정했을 경우, 그 부분만 diff 에 표시됩니다.

![]({{ "/assets/figures/latex_with_github_commit_comment.png" | absolute_url }})

### PDF 파일은 git 외의 수단으로 공유하기

PDF 는 byte 로 기록됩니다. PDF 안에서 업데이트 되는 내용이 git 으로 추적되지는 않습니다. Commit 을 할 때 마다 각 버전을 모두 지니고 있게 될테니 PDF 는 최종본이나 중간 관리본 외에는 git 으로 관리하지 않기로 하였습니다.

### Latex to PDF (compile)

Latex 파일을 local 에서 컴파일해도 좋지만, 몇 번 컴파일 할 일이 없다던지 내용에 대한 수정만 하는 것이라면 online compile service 를 이용하는 것도 편리합니다. [Overleaf](https://www.overleaf.com/) 는 latex 를 PDF 로 컴파일을 해줍니다. 느리긴 하지만, 패키지 설치나 관리의 수고를 덜 수 있습니다.

### Table code generator

Latex 에서 표를 만들 때에는 각 cell 에 대하여 일일히 코딩을 해야 합니다. 혹은 table generator 인 [www.tablesgenerator.com](https://www.tablesgenerator.com/) 를 이용할 수도 있습니다. 특히나 이 사이트는 csv 파일을 읽어들여 표로 만들 수도 있습니다.

## Conclusion

하나의 문서에 대하여 git 으로 문서 버전 관리하는 연습을 해보았는데, 이후로 Word 를 메인으로 쓸 일은 없을 것 같다는 생각이 들었습니다. 쓰던 것들에 대해 판단없이 계속 이용하는 것은 없는지 고민해볼 계기가 되었습니다.

한글 latex 문서의 예시 파일도 [github][latex_sample_git]에 올려두었습니다.

[latex_sample_git]: https://github.com/lovit/latex_sample

### Appendix. 한 줄로 씌여진 문단의 문장을 각 줄로 나누기

이전에 작성한 latex 는 한 문단을 한 줄로 적는 경우가 많았습니다. 문단 단의 문장들을 각 줄로 나누는 설거운 코드를 적어뒀습니다. 고칠 부분은 한 줄이 paragraph 인지 확인하는 부분으로 생각됩니다.

한 줄이 문단이면 이를 문장으로 나눠 쓰는게 목표입니다.

list of str 형태의 lines 를 입력받아 paragraph 인 줄을 separate 함수에 넣어 분리합니다.

{% highlight python %}
def _a_sentence_a_line(lines):
    lines_ = []
    for line in lines:
        if not line.strip():
            lines_.append(line)
        elif line[0] == '%' or line[0] == '\\':
            lines_.append(line)
        elif is_paragraph(line):
            lines_ += separate(line)
        else:
            lines_.append(line)

    return lines_
{% endhighlight %}

is_paragraph 함수는 글자수가 10 보다 크고 두 개 이상의 마침표를 포함하는 줄로 정의하였습니다. 한 문장으로 이뤄진 문단이라면 어자피 그대로 쓰면 되니까요. 

{% highlight python %}
def is_paragraph(line):
    return len(line.strip()) > 10 and line.count('.') > 1

def separate(line):
    l_blank = len(line) - len(line.lstrip())
    r_blank = len(line) - len(line.rstrip())
    separateds = line.strip().split('.')
    separateds = [sep+ ('.' if sep else '') for sep in separateds]
    separateds = [sep.strip() for sep in separateds]
    separateds[0] = ' '*l_blank + separateds[0]
    separateds[-1] = ' '*r_blank + separateds[-1]
    return separateds
{% endhighlight %}

만든 함수를 테스트 합니다.

{% highlight python %}
s = '문장이다. 두번째다. 세번쩨다.  \n\n두번째 단락이다. 두번째의 두번째다.'
{% endhighlight %}

위 문장은 텍스트 파일에 다음의 형태로 기록되어 있습니다.

    문장이다. 두번째다. 세번쩨다.  

    두번째 단락이다. 두번째의 두번째다.

이를 앞서 정의한 함수에 넣습니다.

{% highlight python %}
lines = s.split('\n')
lines_ = _a_sentence_a_line(lines)
{% endhighlight %}

결과는 아래와 같습니다.

    문장이다.
    두번째다.
    세번쩨다.


    두번째 단락이다.
    두번째의 두번째다.

load, save 부분만 추가합니다.

{% highlight python %}
def a_sentence_a_line(inpath, outpath, encoding='utf-8'):
    lines = _load_docs(inpath, encoding)
    lines_ = _a_sentence_a_line(lines)
    _save_docs(lines_, outpath, encoding)

def _load_docs(inpath, encoding):
    # \n 제거를 위해 line[:-1]
    # line.strip() 을 하면 앞의 빈 칸도 사라짐
    with open(inpath, encoding=encoding) as f:
        lines = [line[:-1] for line in f if line]
    return lines

def _save_docs(lines, outpath, encoding):
    with open(outpath, 'w', encoding=encoding) as f:
        for line in lines:
            f.write('%s\n'%line)
{% endhighlight %}