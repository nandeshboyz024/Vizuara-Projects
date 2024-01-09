const Animation =()=>{
        const canvas = document.getElementById('cv');
        let div = document.getElementById('mydiv');
        canvas.width = div.offsetWidth;
        canvas.height = div.offsetHeight;

        let x1 = Math.random()*(canvas.width-50);
        let y1 = Math.random()*(canvas.height-50);
        let x2 = Math.random()*(canvas.width-50);
        let y2 = Math.random()*(canvas.height-50);
        let x3 = Math.random()*(canvas.width-50);
        let y3 = Math.random()*(canvas.height-50);
        let x4 = Math.random()*(canvas.width-50);
        let y4 = Math.random()*(canvas.height-50);
        let x5 = Math.random()*(canvas.width-50);
        let y5 = Math.random()*(canvas.height-50);
        let x6 = Math.random()*(canvas.width-50);
        let y6 = Math.random()*(canvas.height-50);

        function resizeCanvas(){
            canvas.width = div.offsetWidth;
            canvas.height = div.offsetHeight;
        }
        var ctx = canvas.getContext('2d');

        var image1 = new Image();
        var image2 = new Image();
        var image3 = new Image();
        var image4 = new Image();
        var image5 = new Image();
        var image6 = new Image();

        image1.src='/media/joy.gif';
        image2.src='/media/anger.gif';
        image3.src='/media/fear.gif';
        image4.src='/media/love.gif';
        image5.src='/media/sadness.gif';
        image6.src='/media/surprise.gif';

        var speedX1 = 0.3;
        var speedY1 =0.1;
        var speedX2 = 0.3;
        var speedY2 =0.1;
        var speedX3 = 0.3;
        var speedY3 =0.1;
        var speedX4 = -0.3;
        var speedY4 =-0.1;
        var speedX5 = -0.3;
        var speedY5 =-0.1;
        var speedX6 = -0.3;
        var speedY6 =-0.1;
        function draw(){
            ctx.clearRect(0,0,canvas.width,canvas.height);
            ctx.drawImage(image1,x1,y1,40,40);
            ctx.drawImage(image2,x2,y2,40,40);
            ctx.drawImage(image3,x3,y3,40,40);
            ctx.drawImage(image4,x4,y4,40,40);
            ctx.drawImage(image5,x5,y5,40,40);
            ctx.drawImage(image6,x6,y6,40,40);
            x1=(x1+speedX1+canvas.width)%canvas.width;
            y1=(y1+speedY1+canvas.height)%canvas.height;
            x2=(x2+speedX2+canvas.width)%canvas.width;
            y2=(y2+speedY2+canvas.height)%canvas.height;
            x3=(x3+speedX3+canvas.width)%canvas.width;
            y3=(y3+speedY3+canvas.height)%canvas.height;
            x4=(x4+speedX4+canvas.width)%canvas.width;
            y4=(y4+speedY4+canvas.height)%canvas.height;
            x5=(x5+speedX5+canvas.width)%canvas.width;
            y5=(y5+speedY5+canvas.height)%canvas.height;
            x6=(x6+speedX6+canvas.width)%canvas.width;
            y6=(y6+speedY6+canvas.height)%canvas.height;
            requestAnimationFrame(draw);
        }


        window.addEventListener('resize',function(){
            resizeCanvas();
            draw();
        })
        image1.onload = function(){
            canvas.width = div.offsetWidth;
            canvas.height = div.offsetHeight;
            draw();
        }
        image1.onerror = function(){
            console.error('error');
        }
        image2.onload = function(){
            canvas.width = div.offsetWidth;
            canvas.height = div.offsetHeight;
            draw();
        }
        image2.onerror = function(){
            console.error('error');
        }
        image3.onload = function(){
            canvas.width = div.offsetWidth;
            canvas.height = div.offsetHeight;
            draw();
        }
        image3.onerror = function(){
            console.error('error');
        }
        image4.onload = function(){
            canvas.width = div.offsetWidth;
            canvas.height = div.offsetHeight;
            draw();
        }
        image4.onerror = function(){
            console.error('error');
        }
        image5.onload = function(){
            canvas.width = div.offsetWidth;
            canvas.height = div.offsetHeight;
            draw();
        }
        image5.onerror = function(){
            console.error('error');
        }
        image6.onload = function(){
            canvas.width = div.offsetWidth;
            canvas.height = div.offsetHeight;
            draw();
        }
        image6.onerror = function(){
            console.error('error');}
};

export default Animation;