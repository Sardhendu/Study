library(sp)
library(rgdal)

data(state)
dim(state)

states <- data.frame(state.x77, state.center)
states <- states[states$x > -121,]
head(states)
dim(states)

# The 
coordinates(states) <- c("x", "y")
proj4string(states) <- CRS("+proj=longlat +ellps=clrk66")
summary(states)

# This is projecting the lat-lon coordinates on the 
state.merc <- spTransform(states,
                          CRS=CRS("+proj=merc +ellps=GRS80 +units=us-mi"))
summary(state.merc)

# Gives the detail about the CRS used
CRS("+init=epsg:4326") # LatLon cordinate system
CRS("+init=epsg:3395") # Mercator coordiante system (projected into plane map)


# The UTM projection
proj4string(x) <- CRS("+proj=utm +zone=10 +datum=WGS84")


############  Creating a Dummy Dataset   ###########

dataIN = data.frame(c(-80.2, -80.11, -81.0), c(11.1, 11.1345, 11.2))

colnames(dataIN) <- c('longitude', 'logitude')
dataIN

# Now we define a projection (In this case it is a latlong projection)
llCRS <- CRS("+proj=longlat +ellps=WGS84")

# Now we say R about the columns that are spatial points
coords <- SpatialPoints(dataIN[, c('longitude', 'logitude')], proj4string = llCRS)  

# We create a dataframe includint the spatial columns
dataIN_Spatial_DF <- SpatialPointsDataFrame(coords, dataIN)

# Now its time to project the points into a different coordinate system (In this case into a Mercator, epsg:3395)
out <- spTransform(dataIN_Spatial_DF, CRS=CRS("+init=epsg:3395 +proj=merc +ellps=WGS84"))
# To convert them into miles
out_mi <- spTransform(dataIN_Spatial_DF, CRS=CRS("+init=epsg:3395 +proj=merc +ellps=WGS84 +units=us-mi"))

