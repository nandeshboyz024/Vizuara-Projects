import React from 'react'
import './animal-detection/classic.css'
import {moveNext,movePrev,allowDrop,drag,drop} from './animal-detection/function'
import Heading from './Heading'
import {Link} from 'react-router-dom';
export default function Giraffes() {
  return (
    <div>
        <Heading/>
        <div>
            <div style={{display:'flex'}}>
                <div style={{width:'5%',marginTop:'140px',textAlign:'center'}}>
                    <button className='arrow' id="prev" onClick={movePrev}>
                        <img src="imgs/left-arrow.png" alt="left-arrow"  width={20} height={20}/>
                    </button>
                </div>
                <div style={{width:'90%'}}>
                    <div className='container' id="c1" style={{textAlign:'center'}}>
                        <div className='item  item-1' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>angolan-giraffe</p>
                            <img id='img1' src="imgs/giraffes/angolan-giraffe.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-2' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>masai-giraffe</p>
                            <img id='img2' src="imgs/giraffes/masai-giraffe.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className='item  item-3' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>nubian-giraffe</p>
                            <img id='img3' src="imgs/giraffes/nubian-giraffe.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210}/>            
                        </div>
                        <div className='item  item-4' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>reticulated-giraffe</p>
                            <img id='img4' src="imgs/giraffes/reticulated-giraffe.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                        <div className='item  item-5' style={{padding:'10px'}}>
                            <p style={{marginBottom:'0'}}>thornicroft-giraffe</p>
                            <img id='img5' src="imgs/giraffes/thornicroft-giraffe.png" draggable='true' onDragStart={drag} alt="WildEye" width={300} height={210} />            
                        </div>
                    </div>
                </div>
                <div style={{width:'5%',marginTop:'140px',textAlign:'center'}}>
                    <button className='arrow' id="next" onClick={moveNext}>
                        <img src="imgs/right-arrow.png" alt="right-arrow" width={20} height={20} />
                    </button>
                </div>
            </div>
        </div>

       
        <div  style={{padding:'20px'}}>
            <div className='row'>
                <div className='col-lg-2'>
                    <Link to="/wildlife-spotting/zebras" style={{textDecoration:'none'}}>
                        <div className='item' style={{textAlign:'center'}} >
                            <h3>Previous Page</h3>
                            <img src="imgs/logo9.png" alt="logo9"  width={200} height={140}/>
                            <h3 style={{color:'#000000'}}>Zebra</h3>
                        </div>
                    </Link>
                </div>
                <div className='col-lg-8 text-center'>
                    <div id="div1" style={{width:'500px',height:'350px',border:'1px solid black',margin:'auto',display: 'flex',justifyContent: 'center',alignItems: 'center',position: 'relative',background:'lightblue'}} onDrop={drop} onDragOver={allowDrop}>
                        <h4 style={{color:'red'}}>drag and drop</h4>
                        <img src="imgs/drop.png" width={200} height={200} style={{position:'absolute'}} alt="drop here" />
                    </div>
                </div>
                <div className='col-lg-2'>
                    <Link to="/wildlife-spotting/persons" style={{textDecoration:'none'}}>
                        <div className='item' style={{textAlign:'center'}}>
                            <h3>Next Page</h3>
                            <img src="imgs/logo11.png" alt="logo11"  width={200} height={140}/>
                            <h3 style={{color:'#6F973E'}}>Human</h3>
                        </div>
                    </Link>
                </div>
            </div>
        </div>
    </div>
  )
}
