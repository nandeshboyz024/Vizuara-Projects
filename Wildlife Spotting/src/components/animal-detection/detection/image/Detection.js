import React, {  useState, useRef, useContext} from 'react';
import { ModelContext } from '../context/model-context';
import { useDimension } from '../utils/dimension-hook';
import useDetector from './useDetector';
import '../../global.css'
import MagicDropzone from 'react-magic-dropzone';
const Detection = () => {
    
    const { model, labels } = useContext(ModelContext);
    const dimensions = useDimension();
    const [loadedImg, setLoadedImg] = useState(null); 
    const imageRef = useRef();
    const canvasRef = useRef();

    useDetector(model, labels, loadedImg, imageRef, canvasRef)

    const onDrop = (accepted, rejected, links) => {
        if (accepted && accepted.length > 0) {
            const reader = new FileReader();
            reader.addEventListener("load", () => {
                setLoadedImg(reader.result);
            });
            reader.readAsDataURL(accepted[0]);
        }
    }
    return (
        
        <React.Fragment>
                <div>
                    <div style={{width:'60%',margin:'0 auto'}}>
                        <div className='card'>
                            <MagicDropzone className='dropzone' accept="image/jpeg, image/png, .jpg, .jpeg, .png" multiple={false} onDrop={onDrop}>
                                <div className='center-div-wrap'>
                                    {loadedImg ? (
                                        <img className='dropzone-image' src={loadedImg} width="100" alt="drop" />
                                    ) : (
                                        <div>
                                            <div className='center-div'>
                                                <img src='imgs/upload.png' alt="upload" width={50} height={50} srcSet=''/>
                                            </div>
                                            <div style={{ display: 'block' }}>
                                                <p style={{fontSize:'20px'}}>Drag/drop files over here</p>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </MagicDropzone>
                        </div>
                    </div>
                    <div className="center-div">
                        {loadedImg && (
                            <div style={{
                                width: `${dimensions.width}px`,
                                height: `${dimensions.height}px`,
                            }}>
                                <div className="image-container">
                                    <img
                                        src={loadedImg} 
                                        width={dimensions.width}
                                        height={dimensions.height} 
                                        className="image-canvas"
                                        ref={imageRef}
                                        alt='view'
                                    />
                                    <canvas 
                                        width={dimensions.width}
                                        height={dimensions.height} 
                                        className="image-canvas" 
                                        ref={canvasRef}
                                    />
                                </div>
                            </div>
                        )}
                    </div>
                </div>
        </React.Fragment>
    )
}

export default Detection;