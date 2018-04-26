---
title: Plotly 를 이용한 3D scatter plot
date: 2018-04-26 22:00:00
categories:
- visualization
tags:
- visualiztion
---

Plotly 는 Python 에서 data visualization 을 도와주는 패키지입니다. Bokeh 처럼 Java scrpit 기반으로 작동되는 시각화 웹을 Python 에서 이용할 수 있도록 도와줍니다. 3D scatter plot 을 그리기 위해 적절한 패키지를 찾던 중 Plotly 를 써보게 되었습니다. Swiss-roll data 를 이용한 3D scatter plot 의 quick starting 을 정리하였습니다.

## Swiss-roll data and manifold learning

Swiss roll data 는 embedding, manifold learning 에서 단골로 이용되는 인공 실험데이터 입니다. Swiss roll 은 우리가 잘 아는 롤케익입니다. 아래의 롤케익처럼 데이터가 둘둘 말려있는 형태입니다. 3차원 공간에서는 말려있는 형태지만, 우리는 이 롤케익을 펼치면 한 장의 판이 있다는 것을 알고 있습니다. 

![](https://pioneerwoman.files.wordpress.com/2015/12/chocolate-swiss-roll-cake-00.jpg?w=780&h=521)
<center>Source from http://thepioneerwoman.com/food-and-friends/chocolate-swiss-roll-cake/</center>

Manifold 는 국소적으로 Euclidean distance 가 이용될 수 있지만, 전체에서는 이를 적용할 수 없는 공간입니다. 아래 그림의 A, B 는 swiss roll data 입니다. A 그림처럼 3차원의 두 점 사이의 거리는 Euclidean distance 로 표현할 수도 있습니다. 하지만 우리가 생각하는 (심리적) 거리는 Euclidean 이 아닙니다. 그림 B 처럼 롤케익을 따라 한바퀴 돌아간 거리가 두 점 사이의 거리입니다. 즉, 이 데이터에서의 두 점의 거리는 Euclidean 으로 정의되지 않습니다. 하지만 롤케익의 끝부분을 조금 잘라내면 그 부분에서는 Euclidean distance 로 두 점 사이의 거리를 어느 정도 표현할 수 있습니다. 이처럼 국소적으로는 Euclidean distance 가 적용될 수 있지만, 전체 공간에서는 적용될 수 없는 데이터를 Manifold 라 합니다. 

![](http://benalexkeen.com/wp-content/uploads/2017/05/isomap.png)
<center>Source from http://benalexkeen.com/isomap-for-dimensionality-reduction-in-python/</center>

Manifold 의 예시로 자주 이용되는 것은 지표 (Earth surface)에서의 이동거리입니다. 한국과 아르헨티나는 지구 반대편에 있습니다. 3 차원 공간에서 두 나라의 Euclidean distance 는 지구의 지름만큼 입니다. 하지만, 실제 우리가 아르헨티나를 가기 위해서 비행기를 탄다면 지표를 따라 돌아가야 합니다. 지표는 Manifold 공간입니다.

Manifold learning 을 쉽게 설명하면 swiss roll data 처럼 베베꼬인 공간을 말끔하게 펴는 것입니다. 더 정확히는 원 공간에서의 데이터의 구조를 잘 보존하는 저차원의 표현 방법을 학습하는 것입니다. 다시 말해, 3 차원의 swiss roll data 를 2 차원으로 보고 싶은 것입니다. 

대표적으로 Locally Linear Embedding (LLE), ISOMAP, t-Stochastic Neighbor Embedding (t-SNE) 등이 있습니다.

3 차원은 우리가 확인할 수 있는 공간이며, 우리가 기대하는 swiss roll data 의 2 차원의 모습이 명확하기 때문에 manifold learning 에서 swiss roll data 는 자주 이용되었습니다.

## Plotly

오랜만에 swiss roll data 를 다룰 일이 생겼습니다. 이전에는 Matlab 을 이용하여 plotting 을 하였는데, Bokeh 를 배웠으니 이를 이용하고 싶었습니다. Bokeh 의 현재 버전 (0.12.15) 에서는 3D scatter plot 이 공식 API 로 지원되지는 않았습니다. 대신 [extension gallery][bokeh3d] 에 들어가면 Bokeh 의 함수들을 이용한 코드가 제공되고 있습니다. 이왕이면 안정적인 공식 API 를 쓰고 싶어져서 다른 패키지를 찾게 되었습니다. 

Plotly 도 Python 의 data visualization 에서 자주 이용되는 패키지입니다. 이번을 기회로 한 번 써보고 싶어졌습니다.

Plotly 의 설치는 pip install 로 가능합니다. 이 포스트의 작성 당시 버전은 2.5.1 입니다.

    pip install plotly

Plotly 는 작업한 plot 을 Plotly cloud 에 올릴 수 있는 기능이 있습니다. 하지만 저는 제 local 에 그림을 그리는 것이 목적이기 때문에 plotly.offline 을 이용하였습니다. Pip install 이후 offline 을 이용할 수 있습니다. 

## Generate swiss-roll data

Scikit learn 의 data 에서는 인공데이터를 만들 수 있는 함수를 제공해줍니다. Two moon 과 같은 유명한 데이터셋을 만들 수 있습니다. Swiss roll data generator 도 제공합니다. 

make_swiss_roll() 함수의 n_samples 는 데이터의 개수이며, noise 는 pertubation 의 정도입니다. Gaussian distribution 의 standard deviation 입니다. noise 가 크면 두꺼운 롤케익이 되고, noise=0 이면 종잇장같은 롤케익이 만들어집니다. noise 가 너무 크면 공간이 얽히니 몇 번 noise 값을 바꿔가며 실험에 이용할 데이터를 만들면 됩니다.

{% highlight python %}
from sklearn.datasets import make_swiss_roll

X, t = make_swiss_roll(n_samples=1000, noise=0)
{% endhighlight %}

Return 은 X 와 t 가 됩니다. X 는 3 차원의 데이터입니다. t 는 각 데이터의 position 값 입니다. 롤케익의 중심에서부터의 거리라 생각할 수 있습니다.

{% highlight python %}
print(X.shape) # (1000, 3)
print(t.shape) # (1000,)
{% endhighlight %}

{% highlight python %}
print(t[:10])
# array([10.83009033, 13.99799587,  8.88363173, 13.8109314 , 11.43990206,
        7.02263808,  6.10439665, 12.33610539,  5.25671603, 10.00544222])
{% endhighlight %}

t 를 color label 로 이용하려면 [0, 1] 사이의 scale 로 만들어야 합니다. 함수도 있지만, 간단한 코딩으로 color 를 직접 만듭니다.

{% highlight python %}
color = (t - t.min()) / (t.max() - t.min())
{% endhighlight %}

## Plotly 를 이용한 3D scatter plot

Plotly offline 는 plotly.offline 을 이요하면 됩니다. Plotting 에 이용할 data 와 layer 는 plotly.graph_objs 를 이용합니다.

{% highlight python %}
from plotly.offline import plot
import plotly.graph_objs as go
{% endhighlight %}

offline 기능은 1.9.0 이상부터 지원한다고 합니다. offline 의 Jupyter notebook 에서 Plotly 를 이용하려면 다음을 실행해야 합니다. 

{% highlight python %}
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
{% endhighlight %}

데이터는 graph_objs.Scatter3d 를 이용합니다. 

x, y, z 는 3 차원의 그 x, y, z 입니다. 

text 는 각 데이터에 관련된 메모입니다. 만약 영화 리뷰에 관련한 문서 군집화를 수행한 뒤, 이를 2 차원 벡터로 변환하고, 이를 Plotly 를 이용하여 시각화 한다면, 각 영화 리뷰 데이터에 영화 제목을 넣을 수 있습니다. 혹은 영화 리뷰의 snippest 를 넣을 수도 있습니다. Plotly 의 3d scatter plot 은 interactive 한 시각화이기 때문에 데이터의 label 이나 데이터 값을 확인하는 용도로 이용할 수 있습니다. 이 점 때문에 Plotly 의 3d scatter plot 이 확 꽂혔습니다. 

marker 는 각 점에 대한 설정입니다. size 는 점의 크기이며, color 에는 'rbg(217, 217, 217)' 같은 RBG 코드를 넣을 수도 있습니다. 우리는 [0, 1] 사이의 value 를 color 로 이용할 것이기 때문에 color=color 로 정의합니다. 

[0, 1] 사이의 color 값에 따른 colormap 을 이용할 수 있습니다. Colorscale 은 matplotlib 의 colormap, Bokeh 의 Palette 입니다. 

line 은 각 점의 외각선 입니다. 우리는 이용하지 않을 것이기 때문에 width=0 으로 설정하였습니다. 

{% highlight python %}
data = go.Scatter3d(
    x=X[:,0],
    y=X[:,1],
    z=X[:,2],
    text = ['point #{}'.format(i) for i in range(X.shape[0])],
    mode='markers',
    marker=dict(
        size=3,
        color=color,
        colorscale='Jet',
        line=dict(
            #color='rgba(217, 217, 217, 0.14)',
            #color='rgb(217, 217, 217)',
            width=0.0
        ),
        opacity=0.8
    )
)
{% endhighlight %}

graph_objs.Layout 을 이용하여 그림의 layout 을 설정합니다. 

width, height 를 설정합니다. Margin 은 그림의 margin 입니다.

{% highlight python %}
layout = go.Layout(
    autosize=False,
    width=500,
    height=500,
    margin=go.Margin(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    #paper_bgcolor='#7f7f7f',
    #plot_bgcolor='#c7c7c7'
)
{% endhighlight %}

그림은 graph_objs.Figure 를 이용하여 그립니다. 입력되는 arguments 는 list of data 와 layout 입니다. 데이터는 list 형태로 입력합니다. 만약 여러 데이터를 동시에 입력하는 경우라면 아래처럼 list 에 모두 입력하면 됩니다. 

{% highlight python %}
fig = go.Figure(data=[data], layout=layout)
{% endhighlight %}

{% highlight python %}
fig = go.Figure(data=[data1, data2, data3], layout=layout)
{% endhighlight %}

Jupyter notebook 에서 plot 을 보려면 iplot 에 그림을 입력합니다.

{% highlight python %}
iplot(fig)
{% endhighlight %}

그림을 html 파일로 저장하려면 plotly.offline.plot 을 이용합니다.

{% highlight python %}
from plotly.offline import plot
plot(fig, filename='plotly-3d-scatter-small.html', auto_open=False)
{% endhighlight %}

아래는 Plotly 를 이용한 swiss roll 의 3d scatter plot 입니다. 마우스를 점 위에 올리면 points #13 같은 text 가 보여집니다. 

<div id="plotly"></div>

마우스 휠을 이용하면 확대, 축소가 가능합니다. 

사진 아이콘을 누르면 현재의 그림의 snapshot 이 png 로 저장됩니다.

<script src="https://code.jquery.com/jquery-3.2.1.min.js" integrity="sha256-hwg4gsxgFZhOsEEamdOYGBf13FyQuiTwlAQgxVSNgt4=" crossorigin="anonymous"></script>
<script type="text/javascript">
      $(document).ready(function(){
         $("#plotly").load("https://github.com/lovit/lovit.github.io/blob/master/assets/resources/plotly-3d-scatter-small.html")
      });
</script>

[bokeh3d]: http://bokeh.pydata.org/en/latest/docs/user_guide/extensions_gallery/wrapping.html#userguide-extensions-examples-wrapping