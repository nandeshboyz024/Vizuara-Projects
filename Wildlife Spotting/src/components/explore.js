import React from 'react'
import Heading from './Heading'
import './animal-detection/global.css'
import {Link} from 'react-router-dom';
export default function Explore() {
  return (
    <div>
        <Heading/>
        <div style={{paddingTop:'20px'}}>
            <div className='row'>
                <div className='col-lg-7' style={{textAlign:'justify'}}>
                    <h1 style={{marginLeft:'10px'}}>Counting & Monitoring Animals</h1>
                    <p style={{fontSize:'20px',margin:'20px'}}>
                        Animal detection helps in counting and monitoring wildlife populations.
                        Scientists use this AI to keep track of endangered animals, like real-life superheroes for nature.
                        Knowing the population size helps conservationists make informed decisions.
                    </p>
                </div>
                <div className='col-lg-5 text-center'>
                    <video src="demo/counting-animals.mp4" className="img-fluid" width={500} height={350} controls/>
                </div>
            </div>
        </div>
        <div style={{paddingTop:'20px'}}>
            <div className='row'>
                <div className='col-lg-7' style={{textAlign:'justify'}}>
                    <h1 style={{marginLeft:'10px'}}>Why AI is so important for animal safety ?</h1>
                    <p style={{fontSize:'20px',margin:'20px'}}>
                        AI can spot animals from a distance. This helps us know when animals are near the road, giving us time to take action.
                        By alerting drivers and even changing traffic signals, AI can help prevent accidents between animals and vehicles.
                        Some animals are rare or endangered. AI helps us keep track of them and ensures they are protected, allowing us to live together peacefully.
                    </p>
                </div>
                <div className='col-lg-5 text-center'>
                    <video src="demo/elephant-road-crossing.mp4" className="img-fluid" width={500} height={350} controls/>
                </div>
            </div>
        </div>
        <div className='text-center' style={{padding:'20px'}}>
            <Link to="/wildlife-spotting/explore-more" style={{textDecoration:'none'}}>
                <button type="button" style={{background:'linear-gradient(to right, #5170FF,#FF66C4)',textAlign:'center',color:'white',fontWeight:"bold",borderRadius:'10px',padding:'5px'}}>Explore More</button>
            </Link>
        </div>
    </div>
  )
}
