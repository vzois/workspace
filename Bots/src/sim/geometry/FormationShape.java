package sim.geometry;

import java.util.ArrayList;

import math.geom2d.Point2D;
import math.geom2d.conic.Circle2D;
import math.geom2d.line.LineSegment2D;
import math.geom2d.line.Ray2D;
import math.geom2d.polygon.SimplePolygon2D;

public class FormationShape {
	public static int CIRCLE=0;
	public static int DIAMOND=1;
	ArrayList<Point2D> shape;
	int agents;
	
	public FormationShape(int agents){
		this.agents = agents;
		this.shape = new ArrayList<Point2D>(this.agents);
	}
	
	public void createCircle(Point2D center,double radius){
		Ray2D ray;
		Circle2D c2D = new Circle2D(center,radius);
		ArrayList<Point2D> points;
		double alpha = 2*Math.PI/this.agents;
		double angle=0;
		
		for(int i=0;i<this.agents;i++){
			ray = new Ray2D(center,angle);
			points = new ArrayList<Point2D>(c2D.intersections(ray));
			if(!points.isEmpty()) shape.add(points.get(0));
			else System.out.println("Not Intersection Error!!!");
			angle +=alpha;
		}
		//System.out.println("Shape: "+shape);
	}
	
	public void createSquare(Point2D center, double side){
		SimplePolygon2D sp = new SimplePolygon2D();
		ArrayList<Point2D> points;
		
		sp.addVertex(new Point2D(center.x() + side/2, center.y() + side/2));
		sp.addVertex(new Point2D(center.x() - side/2, center.y() + side/2));
		sp.addVertex(new Point2D(center.x() - side/2, center.y() - side/2));
		sp.addVertex(new Point2D(center.x() + side/2, center.y() - side/2));
		
		Ray2D ray;
		double alpha = 2*Math.PI/this.agents;
		double angle=0;
		
		for(int i=0;i<this.agents;i++){
			ray = new Ray2D(center,angle);
				for(LineSegment2D l : sp.edges()){
				points = new ArrayList<Point2D>(l.intersections(ray));
				if(!points.isEmpty()) shape.add(points.get(0));
				//else System.out.println("Not Intersection Error!!!");
			}
			angle +=alpha;
		}
	}
	
	public ArrayList<Point2D> getShape(){
		return this.shape;
	}
}
