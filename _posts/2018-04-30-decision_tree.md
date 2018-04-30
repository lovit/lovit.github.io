---
title: Decision trees are not appropriate for text classifications.
date: 2017-04-30 09:00:00
categories:
- machine learning
tags:
- decision tree
---

의사결정나무 (Decision tree) 는 classification 과정에 대한 해석을 제공하는 점과 다른 classifiers 보다 데이터의 전처리를 (상대적으로) 덜해도 된다는 장점이 있습니다. 하지만 bag of words model 과 같은 sparse data 의 분류에는 적합하지 않습니다. 이번 포스트에서는 의사결정나무가 무엇을 학습하는지 알아보고, 왜 sparse data 에는 적합하지 않은지에 대하여 이야기합니다.

## Decision trees

의사결정나무는 데이터의 공간을 직사각형으로 나눠가며 최대한 같은 종류의 데이터로 이뤄진 부분공간을 찾아가는 classifiers 입니다. 마치 clustering 처럼 비슷한 공간을 하나의 leaf node 로 나눠갑니다.

<div class="slider">
    <img src="/assets/figures/dt_growth_1.png" class="slide" />
    <img src="/assets/figures/dt_growth_2.png" class="slide" />
    <img src="/assets/figures/dt_growth_3.png" class="slide" />
    <img src="/assets/figures/dt_growth_4.png" class="slide" />
    <img src="/assets/figures/dt_growth_5.png" class="slide" />
    <img src="/assets/figures/dt_growth_6.png" class="slide" />
</div>


<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
<script type="text/javascript" >
    $("document").ready(function(){
    var imageArr = document.getElementsByClassName('slide');
    var offset = imageArr.length-1;
    var currentImage, prevImage, nextImage;

    function getCurrentImage() {
        currentImage = imageArr[offset];
    }

    function getPrevImage() {
        if(offset == 0) 
            offset = imageArr.length-1;
        else
            offset = offset-1;

        prevImage = imageArr[offset];

    }

    function getNextImage() {
        if(offset == imageArr.length-1)
            offset = 0;
        else
            offset = offset+1;

        nextImage = imageArr[offset];
    }

    $(".prev").click(function(){

         $(function(){
            getCurrentImage();
         });
         $(function(){
            getPrevImage();
         });

         $(currentImage).css({right: '0px'});
         $(prevImage).css({left: '0px'});

         $(currentImage).animate({width:'80%',width:'60%',width:'40%',width:'20%',width:'0'});
         $(prevImage).animate({width:'20%',width:'40%',width:'60%',width:'80%',width:'100%'});
    });

    $(".next").click(function(){
             $(function(){
                getCurrentImage();
             });
             $(function(){
                getNextImage();
             });


         $(currentImage).css({right: '0px'});
         $(nextImage).css({left: '0px'});

         $(currentImage).animate({width:'80%',width:'60%',width:'40%',width:'20%',width:'0%'});
         $(nextImage).animate({width:'20%',width:'40%',width:'60%',width:'80%',width:'100%'});
    });
 });
</script>

<style>
    .slider {
        width : 90%;
        margin-left: 5%;
        margin-right: 5%;
        height : 400px;
        border : 2px solid black;
        position: relative;
    }
    img {
        width:100%;
        height:400px;
        position: absolute;

    }

    .prev, .next {
        position :relative;
        cursor : pointer;
        width : 4%;
        height: 70px;
        border : 1px solid black;
        margin-top: -250px;
        font-size: 40px;
        color:#fff;
        padding-left:10px;
        background: #000;
        opacity: .5;

    }
    .next {
        float:right;
        margin-right: 0px;

    }
    .prev{
        float:left;
        margin-left: 0px;
    }

    .prev:hover, .next:hover{
        opacity: 1;
    }
</style>