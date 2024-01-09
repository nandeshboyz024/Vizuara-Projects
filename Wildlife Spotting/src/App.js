import React from 'react';
import Home from './components/home';
import { BrowserRouter as Router,Routes,Route} from 'react-router-dom';
import Birds from './components/birds';
import Dogs from './components/dogs';
import Cats from './components/cats';
import Cows from './components/cows';
import Sheeps from './components/sheeps';
import Bears from './components/bears';
import Elephants from './components/elephants';
import Horses from './components/horses';
import Zebras from './components/zebras';
import Giraffes from './components/giraffes';
import Persons from './components/persons';
import Explore from './components/explore';
import ExploreMore from './components/explore-more';
import Detect from './components/detection'

const App = () => {
  return ( 
      <Router>
        <Routes>
          <Route exact path='/wildlife-spotting' element={<Home/>}/>
          <Route exact path="/wildlife-spotting/birds" element={<Birds/>} />
          <Route exact path="/wildlife-spotting/dogs" element={<Dogs/>} />
          <Route exact path="/wildlife-spotting/cats" element={<Cats/>} /> 
          <Route exact path="/wildlife-spotting/cows" element={<Cows/>} /> 
          <Route exact path="/wildlife-spotting/sheeps" element={<Sheeps/>}/>
          <Route exact path="/wildlife-spotting/bears" element={<Bears/>}/>
          <Route exact path="/wildlife-spotting/elephants" element={<Elephants/>}/>  
          <Route exact path="/wildlife-spotting/horses" element={<Horses/>}/>
          <Route exact path="/wildlife-spotting/zebras" element={<Zebras/>}/>
          <Route exact path="/wildlife-spotting/giraffes" element={<Giraffes/>}/>                                    
          <Route exact path="/wildlife-spotting/persons" element={<Persons/>}/>
          <Route exact path="/wildlife-spotting/explore" element={<Explore/>}/>  
          <Route exact path="/wildlife-spotting/explore-more" element={<ExploreMore/>}/>   
          <Route exact path="/wildlife-spotting/detection" element={<Detect/>}/>                                                                 
        </Routes>
      </Router>
  );
}

export default App;
