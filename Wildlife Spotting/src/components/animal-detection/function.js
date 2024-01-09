export const moveNext=()=>{
  const container = document.getElementById('c1');
        if (container) {
          container.scrollBy({
            left: 320, // Adjust the scroll distance as needed
            behavior: 'smooth'
          });
        }
}
export const movePrev=()=>{
  const container = document.getElementById('c1');
      if (container) {
          container.scrollBy({
              left: -320, // Adjust the scroll distance as needed
              behavior: 'smooth'
          });
      }
}
export const allowDrop = (ev) => {
  ev.preventDefault();
}

export  const drag = (ev) => {
  ev.dataTransfer.setData("text", ev.target.id);
}

export const drop = (ev) => {
  ev.preventDefault();
  var data = ev.dataTransfer.getData('text');

  // Create a clone of the dragged element
  var draggedElement = document.getElementById(data);

  // Check if the draggedElement is found
  if (draggedElement) {
    // Remove existing child elements from the container
    var container = document.getElementById('div1');
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    // Create a new Image component
    var clone = new Image();
    clone.src = draggedElement.src;
    clone.width = 500;
    clone.height = 350;

    // Change the image source by replacing ".png" with "1.png"
    var originalSrc = clone.src;
    var newSrc = originalSrc.replace('.png', '1.png');
    clone.src = newSrc;

    // Append the clone to the container
    container.appendChild(clone);
  }
};
