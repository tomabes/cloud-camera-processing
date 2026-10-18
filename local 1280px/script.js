function refreshImages() {
    const timestamp = new Date().getTime();
    document.getElementById("plotImage").src = "people_count_plot.png?" + timestamp;
    document.getElementById("reloadImage").src = "processed_frame.jpg?" + timestamp;
}

setInterval(refreshImages, 1000);
