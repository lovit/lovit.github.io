---
title: Seaborn vs Bokeh: Part 1. Seaborn tutorial
date: 2017-11-22 05:00:00
categories:
- visualization
tags:
- visualization
---

Seaborn 과 Bokeh 는 파이썬에서 이용할 수 있는 plotting 도구들이지만, 둘은 각자 지향하는 목적이 다르며 서로가 더 적합한 상황도 다릅니다. 데이터 분석 결과의 시각화 목적에서 두 패키지가 지원하는 기능을 비교해 봄으로써 각자가 할 수 있는 일과 할 수 없는 일을 알아봅니다. 또한 이 튜토리얼은 두 패키지의 사용법을 빠르게 익히려는 목적에 제작하였습니다. Part 1 은 seaborn 의 사용법이며, official tutorial 를 바탕으로, 알아두면 유용한 이야기들을 추가하고 중복되어 긴 이야기들을 제거하였습니다.

## Plotting with numerical data

Python 으로 plot 을 그릴 때 가장 먼저 생각나는 도구는 matplotlib 입니다. 가장 오래된 패키지이며, 아마도 현재까지는 가장 널리 이용되고 있는 패키지라 생각됩니다. 하지만 matplotlib 은 그 문법이 복잡하고 arguments 이름들이 직관적이지 않아 그림을 그릴때마다 메뉴얼을 찾게 됩니다. 그리고 매번 그림을 그릴 때마다 몇 줄의 코드를 반복하여 작성하게 됩니다. Seaborn 은 이러한 과정을 미리 정리해둔, matplotlib 을 이용하는 high-level plotting package 입니다.

이 튜토리얼은 Seaborn=0.9.0 의 [official tutorial](https://seaborn.pydata.org/tutorial.html) 을 바탕으로, 추가적으로 알아두면 유용한 몇 가지 설명들을 더하였습니다. 기본적인 흐름과 예시는 official tutorials 을 참고하였습니다. 

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

Seaborn 은 Pandas 와 궁합이 좋습니다. Pandas.DataFrame 의 plot 함수는 기본값으로 matplotlib 을 이용합니다. 그리고 seaborn 은 DataFrame 을 입력받아 plot 을 그릴 수 있도록 구현되어 있습니다. Seaborn 에서 제공하는 `tips` 데이터를 이용하여 몇 가지 plots 을 그려봅니다.

```python
tips = sns.load_dataset("tips")
tips.head(5)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



### Scatter plots

두 변수 간의 관계를 살펴볼 수 있는 대표적인 plots 으로는 scatter plot 과 line plot 이 있습니다. 우선 scatter plot 을 그리는 연습을 통하여 seaborn 의 기본적인 문법을 익혀봅니다.

`seaborn.scatterplot()` 에 tips 데이터를 이용한다는 의미로 `data=tips` 를 입력합니다. 이 중 `x` 로 'total_bill' 을, `y` 로 'tip' 을 이용하겠다고 입력합니다. 그러면 그림이 그려집니다.


```python
sns.scatterplot(x="total_bill", y="tip", data=tips)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_5_1.png" | absolute_url }}){: width="50%" height="50%"}

흡연 유무에 따라 서로 다른 색을 칠할 수도 있습니다. 이는 `hue` 에 어떤 변수를 기준으로 다른 색을 칠할 것인지 입력하면 됩니다.

```python
sns.scatterplot(x="total_bill", y="tip", hue="smoker", data=tips)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_7_1.png" | absolute_url }}){: width="50%" height="50%"}

해당 변수값의 종류가 다양할 경우 각 종류별로 서로 다른 색이 칠해집니다.

```python
sns.scatterplot(x="total_bill", y="tip", hue="day", data=tips)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_9_1.png" | absolute_url }}){: width="50%" height="50%"}

`hue` 에 입력되는 값이 명목형이 아닌 실수형일 경우, 그라데이션 형식으로 색을 입력해줍니다.

```python
sns.scatterplot(x="total_bill", y="tip", hue="total_bill", data=tips)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_11_1.png" | absolute_url }}){: width="50%" height="50%"}

Marker style 도 변경이 가능합니다. `style` 에 변수 이름을 입력하면 해당 변수 별로 서로 다른 markers 를 이용합니다. 이후 seaborn 의 style 에 대하여 알아볼텐데, `scatterplot()` 에서의 `style` argument 는 marker style 을 의미합니다.

```python
sns.scatterplot(x="total_bill", y="tip", hue="smoker", style="smoker", data=tips)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_13_1.png" | absolute_url }}){: width="50%" height="50%"}

Marker 의 크기도 조절이 가능합니다.

```python
sns.scatterplot(x="total_bill", y="tip", size="size", data=tips)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_15_1.png" | absolute_url }}){: width="50%" height="50%"}

또한 marker 크기의 상한과 하한도 설정할 수 있습니다.

```python
sns.scatterplot(x="total_bill", y="tip", size="size", sizes=(15, 200), data=tips)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_17_1.png" | absolute_url }}){: width="50%" height="50%"}


`alpha` 는 투명도입니다 (0, 1] 사이의 값을 입력합니다.


```python
sns.scatterplot(x="total_bill", y="tip", size="size", sizes=(15, 200), alpha=.3, data=tips)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_19_1.png" | absolute_url }}){: width="50%" height="50%"}


### relplot() vs scatterplot()

그리고 official tutorial 에서는 `seaborn.relplot()` 함수를 이용하여 이를 그릴 수 있다고 설명합니다. 그런데, 그 아래에 `relplot()` 은 `FacetGrid` 와 `scatterplot()` 과 `lineplot()` 의 혼합이라는 설명이 있습니다. 아직 우리가 한 번에 여러 장의 plots 을 그리는 일이 없었기 때문에 `scatterplot()` 과 `relplot()` 의 차이가 잘 느껴지지는 않습니다. 하지만 `scatterplot()` 에서 제공하는 모든 기능은 `relplot()` 에서 모두 제공합니다. 다른 점은 `relplot()` 은 `scatterplot()` 과 `lineplot()` 을 모두 호출하는 함수입니다. 어떤 함수를 호출할 지 `kind` 에 정의해야 합니다. 즉 `relplot(kind='scatter')` 를 입력하면 이 함수가 `scatterplot()` 함수를 호출합니다. `kind` 의 기본값은 scatter 이므로, scatter plot 을 그릴 때에는 이 값을 입력하지 않아도 됩니다. 

한 장의 scatter/line plot 을 그릴 때에도 `relplot()` 은 이용가능하기 때문에 이후로는 특별한 경우가 아니라면 `relplot()` 을 이용하도록 하겠습니다.

```python
sns.relplot(x="total_bill", y="tip", hue="smoker", size="size", sizes=(15, 200), data=tips, kind='scatter')
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_21_1.png" | absolute_url }}){: width="50%" height="50%"}


그런데 `seaborn.relplot()` 함수를 실행시키면 그림이 그려진 것과 별개로 다음과 같은 글자가 출력됩니다. at 뒤의 글자는 함수를 실행할때마다 달라집니다. 이는 `relplot()` 함수가 return 하는 변수 설명으로, at 뒤는 메모리 주소입니다.

```
<seaborn.axisgrid.FacetGrid at 0x7f0dfda88278>
```

그리고 `seaborn.scatterplot()` 함수의 return 에는 FacetGrid 가 아닌 AxesSubplot 임도 확인할 수 있습니다. FacetGrid 는 1개 이상의 AxesSubplot 의 모음입니다. `seaborn.scatterplot()` 과 `seaborn.lineplot()` 은 한 장의 matplotlib Figure 를 그리는 것이며, `relplot()` 은 이들의 묶음을 return 한다는 의미입니다. 이 의미는 뒤에서 좀 더 알아보도록 하겠습니다. 중요한 점은 두 함수가 각각 무엇인가를 return 한다는 것입니다.

```
<matplotlib.axes._subplots.AxesSubplot at 0x7f0dfd9c7358>
```

이 return 된 변수를 이용하여 그림을 수정할 수 있습니다. 이제부터 변수가 return 됨을 명시적으로 표현하기 위하여 `seaborn.relplot()` 이나 `seaborn.scatterplot()` 을 실행한 뒤, 그 값을 변수 `g` 로 받도록 하겠습니다.

## Utils
### Title

대표적인 수정 작업 중 하나는 그림의 제목을 추가하는 것입니다. 위의 그림에 제목을 추가해봅니다.


```python
g = sns.relplot(x="total_bill", y="tip", hue="smoker", size="size", sizes=(15, 200), data=tips, kind='scatter')
g = g.set_titles('scatter plot example')
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_24_0.png" | absolute_url }}){: width="50%" height="50%"}

그런데 어떤 경우에는 (이유를 파악하지 못했습니다) 제목이 추가되지 않습니다. 이 때는 아래의 코드를 실행해보세요. Matplotlib 은 가장 최근의 그림 위에 plots 을 덧그립니다. 아래 코드는 이미 그려진 g 위에 제목을 추가하는 코드입니다.

```python
g = sns.relplot(x="total_bill", y="tip", hue="smoker", size="size", sizes=(15, 200), data=tips, kind='scatter')
plt.title('scatter plot example')
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_26_1.png" | absolute_url }}){: width="50%" height="50%"}

### Save

`relplot()` 함수를 실행할 때마다 새로운 그림을 그리기 때문에 이들을 변수로 만든 뒤, 각각 추가작업을 할 수 있습니다. 그 중 하나로 그림을 저장할 수 있습니다. 두 종류의 그림을 `g0`, `g1` 으로 만든 뒤, 각 그림을 `savefig` 함수를 이용하여 저장합니다. 저장된 그림을 살펴봅니다.

참고로 FacetGrid 는 `savefig` 기능을 제공하지만, AxesSubplot 은 이 기능을 제공하지 않습니다. 물론 `matplotlib.pyplot.savefig()` 함수나 `get_figure().savefig()` 함수를 이용하면 되지만, 코드가 조금 길어집니다. 이러한 측면에서도 `scatterplot()` 보다 `relplot()` 을 이용하는 것이 덜 수고스럽습니다.


```python
g0 = sns.relplot(x="total_bill", y="tip", hue="smoker", size="size", data=tips, kind='scatter')
g1 = sns.relplot(x="total_bill", y="tip", size="size", sizes=(15, 200), data=tips)

g0.savefig('total_bill_tip_various_color_by_size.png')
g1.savefig('total_bill_tip_various_size_by_size.png')
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_28_0.png" | absolute_url }}){: width="50%" height="50%"}

![]({{ "/assets/figures/seaborn_tutorial_00/output_28_1.png" | absolute_url }}){: width="50%" height="50%"}


### Pandas.DataFrame.plot

Pandas 의 DataFrame 에는 손쉽게 plot 을 그리는 함수가 구현되어 있습니다. kind 에 plot 의 종류를, x, y, 그 외의 title 과 같은 attributes 를 keywords argument 형식으로 입력할 수 있습니다. 그런데 DataFrame 의 plot 함수의 return type 은 Figure 가 아닌, `AxesSubplot` 입니다. 앞서 언급한 것처럼 `AxesSubplot` 은 그림의 저장 기능을 직접 제공하지 않습니다. 대신 `AxesSubplot.get_figure()` 를 이용하여 Figure 를 만들면 `savefig` 를 이용할 수 있습니다.

```python
g = tips.plot(x='total_bill', y='tip', kind='scatter', title='pandas plot example')
g = g.get_figure()
g.savefig('pandas_plot_example.png')
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_30_0.png" | absolute_url }}){: width="50%" height="50%"}

혹은 matplotlib.pyplot.savefig 를 이용하여 AxesSubplot 상태에서 바로 저장할 수도 있습니다.

또한 위에서 return 을 변수로 받지 않고도 그림을 저장하였는데, 이는 matplotlib 은 어떤 그림을 저장할 것인지 설정하지 않으면 가장 마지막에 그린 그림에 대하여 저장을 수행합니다. 그런데 이런 코드는 혼동이 될 수 있기 때문에 코드가 한 줄 더 길어지지만, 저는 return type 을 명시하는 위의 방식을 선호합니다.

```python
ax = tips.plot(x='total_bill', y='tip', kind='scatter', title='pandas plot example')
plt.savefig('pandas_plot_example_2.png')
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_32_0.png" | absolute_url }}){: width="50%" height="50%"}

### matplotlib.pyplot.close()

`seaborn.relplot()` 을 두 번 이용할 경우 각각의 그림이 그려졌습니다. 그런데 `seaborn.scatterplot()` 을 실행하면 두 그림이 겹쳐져 그려집니다. 이를 알아보기 위하여 random noise data 를 만들었습니다.


```python
data = {
    'x': np.random.random_sample(100) * 50,
    'y': np.random.random_sample(100) * 10
}
random_noise_df = pd.DataFrame(data, columns=['x', 'y'])
random_noise_df.head(5)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>46.181098</td>
      <td>1.073238</td>
    </tr>
    <tr>
      <th>1</th>
      <td>19.155420</td>
      <td>6.603210</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32.797057</td>
      <td>3.273879</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33.897212</td>
      <td>3.974610</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.294968</td>
      <td>5.602740</td>
    </tr>
  </tbody>
</table>
</div>


각각의 데이터를 `seaborn.scatterplot()` 에 넣으니 두 그림이 겹쳐져 그려집니다.


```python
g0 = sns.scatterplot(x="total_bill", y="tip", hue='smoker', alpha=0.8, size="size", sizes=(15, 200), data=tips)
g1 = sns.scatterplot(x="x", y="y", alpha=0.2, color='g', data=random_noise_df)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_36_0.png" | absolute_url }}){: width="50%" height="50%"}

실제로 g0, g1 의 메모리 주소가 같습니다.

```python
g0, g1
```

```
(<matplotlib.axes._subplots.AxesSubplot at 0x7fc7ead8ff98>,
 <matplotlib.axes._subplots.AxesSubplot at 0x7fc7ead8ff98>)
```

이 경우, 두 그림을 다르게 그리기 위해서는 `matplotlib.pyplot.close()` 함수를 중간에 실행시켜야 합니다. Matplotlib 은 현재의 Figure 가 닫히지 않으면 계속 그 Figure 위에 그림을 덧그리는 형식입니다. 그래서 앞서 `matplotlib.pyplot.title()` 함수를 실행하여 제목을 더할 수도 있었습니다. 즉 그림이 계속 수정된다는 의미입니다.

```python
g0 = sns.scatterplot(x="total_bill", y="tip", hue='smoker', alpha=0.8, size="size", sizes=(15, 200), data=tips)
plt.close()
g1 = sns.scatterplot(x="x", y="y", alpha=0.2, color='g', data=random_noise_df)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_40_0.png" | absolute_url }}){: width="50%" height="50%"}

그래서 중간에 `close()` 를 실행한 경우에는 각각의 그림에 대하여 제목을 추가하여 Figure 로 만들 수 있습니다.

```python
g0.set_title('total bill ~ tip scatter plot')
g0.get_figure()
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_42_0.png" | absolute_url }}){: width="50%" height="50%"}

```python
g1.set_title('random noise')
g1.get_figure()
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_43_0.png" | absolute_url }}){: width="50%" height="50%"}

이 때는 메모리 주소가 다릅니다.


```python
g0, g1
```

```
(<matplotlib.axes._subplots.AxesSubplot at 0x7fc7ead7a630>,
 <matplotlib.axes._subplots.AxesSubplot at 0x7fc7eacd00b8>)
```

그럼 언제 `matplotlib.pyplot.close()` 가 실행될까요? `relplot()` 이 다시 호출될 때 이전에 그리던 Figure 를 닫고, 새 Figure 를 그리기 시작합니다.

```python
g0 = sns.relplot(x="total_bill", y="tip", hue='smoker', alpha=0.8, size="size", sizes=(15, 200), data=tips)
g1 = sns.scatterplot(x="x", y="y", alpha=0.2, color='g', data=random_noise_df)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_47_0.png" | absolute_url }}){: width="50%" height="50%"}

그래서 `seaborn.scatterplot()` 을 실행한 뒤 `seaborn.relplot()` 을 실행하면 그림이 분리되어 그려집니다. 혼동될 수 있으니 새 그림이 그려질 때에는 습관적으로 `close()` 함수를 호출하는 것이 명시적입니다.

```python
g0 = sns.scatterplot(x="total_bill", y="tip", hue='smoker', alpha=0.8, size="size", sizes=(15, 200), data=tips)
g1 = sns.relplot(x="x", y="y", alpha=0.2, color='g', data=random_noise_df)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_49_0.png" | absolute_url }}){: width="50%" height="50%"}

![]({{ "/assets/figures/seaborn_tutorial_00/output_49_1.png" | absolute_url }}){: width="50%" height="50%"}


## Plotting with numerical data 2
### Line plots

데이터가 순차적 형식일 경우 line plot 은 경향을 확인하는데 유용합니다. 우리는 임의의 시계열 데이터를 만들어 line plot 을 그려봅니다. `cumsum()` 함수는 지금까지의 모든 값을 누적한다는 의미입니다. 자연스러운 순차적 흐름을 지닌 데이터가 만들어질 겁니다.

```python
data = {
    'time': np.arange(500),
    'value': np.random.randn(500).cumsum()
}
df = pd.DataFrame(data)
df.head(5)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.207275</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-1.485134</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>-1.718115</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>-1.624952</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-1.555609</td>
    </tr>
  </tbody>
</table>
</div>


`seaborn.lineplot()` 을 이용하여 `x` 와 `y` 축에 어떤 변수를 이용할지 정의합니다.

```python
g = sns.lineplot(x="time", y="value", data=df)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_53_0.png" | absolute_url }}){: width="50%" height="50%"}

이는 `relplot()` 에서 `kind` 를 'line' 으로 정의하는 것과 같습니다. 물론 return type 은 앞서 언급한것처럼 다릅니다.

```python
g = sns.relplot(x="time", y="value", kind="line", data=df)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_55_0.png" | absolute_url }}){: width="50%" height="50%"}

위 데이터는 x 를 중심으로 데이터가 정렬된 경우입니다. 그런데 때로는 데이터가 정렬되지 않은 경우도 있습니다. 이를 위하여 임의의 2 차원 데이터 500 개를 생성합니다.

```python
data = np.random.randn(500, 2).cumsum(axis=0)
df = pd.DataFrame(data, columns=["x", "y"])
df.head(5)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.568999</td>
      <td>-0.805323</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-3.756781</td>
      <td>-0.687732</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.478935</td>
      <td>0.032900</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.896840</td>
      <td>-0.350074</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.633141</td>
      <td>-0.623276</td>
    </tr>
  </tbody>
</table>
</div>


`x` 를 기준으로 정렬되지 않았기 때문에 마치 좌표 위를 이동하는 궤적과 같은 line plot 이 그려졌습니다.

```python
g = sns.relplot(x="x", y="y", sort=False, kind="line", data=df)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_59_0.png" | absolute_url }}){: width="50%" height="50%"}

이를 x 축 기준으로 정렬하여 그리려면 `sort=True` 로 설정하면 됩니다. 시계열 형식의 데이터의 경우, 안전한 plotting 을 위하여 `sort` 는 기본값이 True 로 정의되어 있습니다.

```python
g = sns.relplot(x="x", y="y", sort=True, kind="line", data=df)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_61_1.png" | absolute_url }}){: width="50%" height="50%"}


### Aggregation and representing uncertainty

`seaborn.lineplot()` 의 장점 중 하나는 신뢰 구간 (confidence interval) 과 추정 회귀선 (estminated line) 을 손쉽게 그려준다는 점입니다. 이를 알아보기 위하여 fMRI 데이터를 이용합니다. 이 데이터는 각 사람 (subject) 의 활동 종류 (event) 에 따라 각 시점 (timepoint) 별로 fMRI 의 측정값 중 하나의 센서값을 정리한 시계열 데이터입니다.

```python
fmri = sns.load_dataset("fmri")
fmri.head(5)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>subject</th>
      <th>timepoint</th>
      <th>event</th>
      <th>region</th>
      <th>signal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>s13</td>
      <td>18</td>
      <td>stim</td>
      <td>parietal</td>
      <td>-0.017552</td>
    </tr>
    <tr>
      <th>1</th>
      <td>s5</td>
      <td>14</td>
      <td>stim</td>
      <td>parietal</td>
      <td>-0.080883</td>
    </tr>
    <tr>
      <th>2</th>
      <td>s12</td>
      <td>18</td>
      <td>stim</td>
      <td>parietal</td>
      <td>-0.081033</td>
    </tr>
    <tr>
      <th>3</th>
      <td>s11</td>
      <td>18</td>
      <td>stim</td>
      <td>parietal</td>
      <td>-0.046134</td>
    </tr>
    <tr>
      <th>4</th>
      <td>s10</td>
      <td>18</td>
      <td>stim</td>
      <td>parietal</td>
      <td>-0.037970</td>
    </tr>
  </tbody>
</table>
</div>


`lineplot()` 의 기본값은 신뢰 구간과 추정 회귀선을 함께 그리는 것입니다. 아래 그림은 subject 와 event 의 구분 없이 timepoint 별로 반복적으로 관측된 값을 바탕으로 그려진, 신뢰 구간과 추정 회귀선 입니다.

```python
g = sns.relplot(x="timepoint", y="signal", kind="line", data=fmri)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_65_1.png" | absolute_url }}){: width="50%" height="50%"}

신뢰 구간을 제거하기 위해서는 `ci` 를 None 으로 설정합니다. `ci` 는 confidence interval 의 약자입니다. 하지만 추정된 회귀선은 그려집니다.

```python
g = sns.relplot(x="timepoint", y="signal", kind="line", data=fmri, ci=None)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_67_0.png" | absolute_url }}){: width="50%" height="50%"}

혹은 데이터의 표준 편차를 이용하여 confidence interval 을 그릴 수도 있습니다.

```python
g = sns.relplot(x="timepoint", y="signal", kind="line", data=fmri, ci="sd")
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_69_0.png" | absolute_url }}){: width="50%" height="50%"}

혹은 bootstrap sampling (복원 반복 추출) 을 이용하여 50 % 의 값을 confidence interval 로 이용할 경우에는 `ci=50` 을 입력합니다. 이 때 boostrap sampling 의 개수도 `n_boot`  에서 설정할 수 있습니다.

```python
g = sns.relplot(x="timepoint", y="signal", kind="line", data=fmri, ci=50, n_boot=5000)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_71_0.png" | absolute_url }}){: width="50%" height="50%"}

추정 회귀선은 `estimator` 를 None 으로 설정하면 제거됩니다. 기본 추정 방법은 x 를 기준으로 moving windowing 을 하는 것입니다. 추정선이 없다보니 주파수처럼 signal 값이 요동치는 모습을 볼 수 있습니다.

```python
g = sns.relplot(x="timepoint", y="signal", kind="line", data=fmri, estimator=None)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_73_0.png" | absolute_url }}){: width="50%" height="50%"}


### Add conditions to line plot

`seaborn.lineplot()` 도 `seaborn.scatterplot()` 처럼 `hue` 와 `style` 을 설정할 수 있습니다.

```python
g = sns.relplot(x="timepoint", y="signal", hue="event", kind="line", data=fmri)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_75_0.png" | absolute_url }}){: width="50%" height="50%"}

```python
g = sns.relplot(x="timepoint", y="signal", hue="event", style="event", kind="line", data=fmri)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_76_0.png" | absolute_url }}){: width="50%" height="50%"}

혹은 `hue` 와 `style` 을 다른 기준으로 정의하거나, 선 중간에 x 의 밀도에 따라 marker 를 입력할 수도 있습니다.

```python
g = sns.relplot(x="timepoint", y="signal", hue="region", style="event",
    markers=True, kind="line", data=fmri)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_78_0.png" | absolute_url }}){: width="50%" height="50%"}

혹은 선의 색은 'region' 에 따라 구분하지만, 각 선은 'subject' 를 기준으로 중복으로 그릴 경우 `units` 에 'subject' 를 입력합니다. 만약 `units` 을 설정하면 이때는 반드시 `estimator=None` 으로 설정해야 합니다. 여러 개의 'subject' 가 존재하다보니 선이 지저분하게 겹칩니다.

```python
g = sns.relplot(x="timepoint", y="signal", hue="region",
    units="subject", estimator=None, kind="line",
    data=fmri.query("event == 'stim'"))
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_80_0.png" | absolute_url }}){: width="50%" height="50%"}


### Plotting with date data

시계열 형식의 데이터 중 하나는 x 축이 날짜 형식인 데이터입니다.

```python
data = {
    'time': pd.date_range("2017-1-1", periods=500),
    'value': np.random.randn(500).cumsum()
}
df = pd.DataFrame(data)
df.head(5)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-01-01</td>
      <td>-0.641428</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-01-02</td>
      <td>0.324469</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-01-03</td>
      <td>0.732299</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-01-04</td>
      <td>-1.069557</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-01-05</td>
      <td>-2.109998</td>
    </tr>
  </tbody>
</table>
</div>


이 역시 `seaborn.lineplot()` 을 이용하여 손쉽게 그릴 수 있습니다. 추가로 `autofmt_xdate()` 함수를 이용하면 x 축의 날짜가 서로 겹치지 않게 정리를 도와줍니다.

```python
g = sns.relplot(x="time", y="value", kind="line", data=df)
g.fig.autofmt_xdate()
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_84_0.png" | absolute_url }}){: width="50%" height="50%"}


## multiple plots

앞서 `seaborn.scatterplot()` 과 `seaborn.relplot()` 의 return type 이 각각 `AxesSubplot` 과 `FacetGrid` 로 서로 다름을 살펴보았습니다. `seaborn.relplot()` 의 장점은 여러 장의 plots 을 손쉽게 그린다는 점입니다. 각 'subject' 별로 line plot 을 그려봅니다. 이때는 col 을 'subject' 로 설정한 뒤, col 의 최대 개수를 `col_wrap` 에 설정합니다. 'subject' 의 개수가 이보다 많으면 다음 row 에 이를 추가합니다. 몇 가지 유용한 attributes 도 함께 설정합니다. aspect 는 각 subplot 의 세로 대비 가로의 비율입니다. 세로:가로가 4:3 인 subplots 이 그려집니다. 그리고 세로의 크기는 `height` 로 설정할 수 있습니다.

```python
g = sns.relplot(x="timepoint", y="signal", hue="event", style="event",
    col="subject", col_wrap=5, height=3, aspect=.75, linewidth=2.5,
    kind="line", data=fmri.query("region == 'frontal'"))
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_86_0.png" | absolute_url }}){: width="50%" height="50%"}

그런데 위의 그림에서 `col` 의 값이 정렬된 순서가 아닙니다. 순서를 정의하지 않으면 데이터에 등장한 순서대로 이 값이 그려집니다. 이때는 사용자가 `col_order` 에 원하는 값을 지정하여 입력할 수 있습니다. `row` 역시 `row_order` 를 제공하니, `row` 단위로 subplots 을 그릴 때는 이를 이용하면 됩니다.


```python
col_order = [f's{i}' for i in range(14)]

g = sns.relplot(x="timepoint", y="signal", hue="event", style="event",
    col="subject", col_wrap=5, height=3, aspect=.75, linewidth=2.5,
    kind="line", data=fmri.query("region == 'frontal'"),
    col_order=col_order
)
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_88_0.png" | absolute_url }}){: width="50%" height="50%"}

이는 scatter plot 에도 적용할 수 있습니다. 예를 들어 column 은 변수 'time' 에 따라 서로 다르게 scatter plot 을 그릴 경우, 다음처럼 `col` 에 'time' 을 입력합니다. `hue`, `size` 와 같은 설정은 공통으로 적용됩니다.

```python
g = sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips, col="time")
```

![]({{ "/assets/figures/seaborn_tutorial_00/output_90_0.png" | absolute_url }}){: width="50%" height="50%"}

`row` 를 성별 기준으로 정의하면 (2,2) 형식의 grid plot 이 그려집니다. 그런데 plot 마다 (sex, time) 이 모두 기술되니 title 이 너무 길어보입니다. 이후 살펴볼 FacetGrid 에서는 margin_title 을 이용하여 깔끔하게 col, row 의 기준을 표시하는 방법이 있습니다. 아마 0.9.0 이후의 버전에서는 언젠가 `seaborn.relplot()` 에도 그 기능이 제공되지 않을까 기대해봅니다.

```python
g = sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips, col="time", row="sex")
```

