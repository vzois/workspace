package sim.formation;

import java.util.ArrayList;

import math.geom2d.Point2D;
import sim.structures.Message;

public class ShapeMessage extends Message {
	ArrayList<Point2D> shape;
	
	public ShapeMessage(){
		super();
	}
	
	public void setShape(ArrayList<Point2D> shape){
		this.shape = new ArrayList<Point2D>(shape);
	}
	
	public ArrayList<Point2D> getShape(){
		return this.shape;
	}

}
