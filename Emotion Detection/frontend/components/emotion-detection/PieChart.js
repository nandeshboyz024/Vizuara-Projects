// BarChart.js
import React, { useEffect } from 'react';
import { Chart} from 'chart.js/auto'; // Import from 'chart.js/auto'

const MakeChart = (props) => {
  useEffect(() => {
    // Data and configuration for the bar chart
    
    const data = {
      labels: ['joy', 'sadness', 'anger', 'fear', 'love','surprise'],
      datasets: [
        {
          label: 'Emotion-Density',
          data: props.data,
          backgroundColor: ['lime','blue','red','green','#ff77ff','gold'],
          borderColor: 'rgba(75,192,192,1)',
          borderWidth: 1,
        },
      ],
    };
    const config = {
      type: 'pie',
      data: data,
      options:{
        responsive:false,
        layout:{
            padding:{
                left:20,
                right:0,
                top:30,
                bottom:0,
            },
        },
        animation:{
            duration:5000,
            easing:"easeInOutBounce",
        },
      },
    };
    var myBarChart = new Chart(props.chartId, config);
    return () => {
      myBarChart.destroy();
    };
  }, [props.data,props.chartId]);
  return <canvas id={props.chartId} width="400" height="300"></canvas>;
};

export default MakeChart;
