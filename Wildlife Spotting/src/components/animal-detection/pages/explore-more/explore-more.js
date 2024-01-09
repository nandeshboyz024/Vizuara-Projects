import React from 'react'
import Heading from '../../../components/animal-detection/heading'
export default function ExploreMore() {
  return (
    <div>
        <Heading/>
        <div className='container' style={{paddingTop:'20px'}}>
            <div className='row'>
                <div className='col-lg-4 text-center'>
                    <p>Giraffe and Zebra</p>
                    <video src="/demo/giraffe-and-zebra.mp4" className="img-fluid" width={500} height={350} controls/>
                </div>
                <div className='col-lg-4 text-center'>
                    <p>Duck quaking on a lack</p>
                    <video src="/demo/duck-quaking-on-a-lack.mp4" className="img-fluid" width={500} height={350} controls/>
                </div>
                <div className='col-lg-4 text-center'>
                    <p>Elephant road crossing</p>
                    <video src="/demo/elephant-road-crossing.mp4" className="img-fluid" width={500} height={350}  controls/>
                </div>
            </div>
            <div className='row' style={{paddingTop:'20px'}}>
                
                <div className='col-lg-4 text-center'>
                    <p>Hide and Seek</p>
                    <video src="/demo/hide-and-seek.mp4" className="img-fluid" width={500} height={350} controls/>
                </div>
                <div className='col-lg-4 text-center'>
                    <p>Dog try for helping cross the road</p>
                    <video src="/demo/dog-try-to-helping-cross-road.mp4" className="img-fluid" width={500} height={350} controls/>
                </div>
                <div className='col-lg-4 text-center'>
                    <p>Greatest save by the dog</p>
                    <video src="/demo/greatest-save-by-dog.mp4" className="img-fluid" width={500} height={350} controls/>
                </div>
            </div>
            <div className='row' style={{paddingTop:'20px'}}>
                <div className='col-lg-2'></div>
                <div className='col-lg-4 text-center'>
                    <p>Greatest save by the elephant</p>
                    <video src="/demo/greatest-save-by-elephant.mp4" className="img-fluid" width={500} height={350} controls/>
                </div>
                <div className='col-lg-4 text-center'>
                    <p>Greatest save by the horse</p>
                    <video src="/demo/greatest-save-by-horse.mp4" className="img-fluid" width={500} height={350} controls />
                </div>
                <div className='col-lg-2'></div>
            </div>
        </div>
      
    </div>
  )
}
