package sim.geometry;

import java.awt.Graphics2D;
import java.util.ArrayList;

import sim.structures.Vector;
import math.geom2d.Point2D;
import math.geom2d.line.Line2D;
import math.geom2d.line.LineSegment2D;
import math.geom2d.line.Ray2D;
import math.geom2d.polygon.SimplePolygon2D;

public class Trapezoid {
	protected SimplePolygon2D convexHull;
	protected CircularSector cs;
	protected ArrayList<LineSegment2D> thisLines;
	
	public Trapezoid(CircularSector cs){
		this.cs = cs;
	}
	
	public void updateTrapezoid(Vector apex){
		ArrayList<Point2D> points = new ArrayList<Point2D>();
		for(Point2D p : this.convexHull.vertices()){
			points.add(p.plus(apex));
		}
		this.convexHull = new SimplePolygon2D(points);
	}
	
	public void createTrapezoid(double selfSize,double obstacleSize){
		SimplePolygon2D A = this.cs.getPolygon();		
		SimplePolygon2D B = new SimplePolygon2D();
		
		double size = obstacleSize/2 + selfSize/2;
		if(this.cs.center.getY()>this.cs.leftPoint.getY()){
			B.addVertex(new Point2D(-size,-size));
			B.addVertex(new Point2D(size,-size));
			B.addVertex(new Point2D(size,size));
			B.addVertex(new Point2D(-size,size));
		}else{
			B.addVertex(new Point2D(size,size));
			B.addVertex(new Point2D(-size,size));
			B.addVertex(new Point2D(-size,-size));
			B.addVertex(new Point2D(size,-size));
		}

		MinkowskiSum ms = new MinkowskiSum(A,B);
		ms.sum();
		this.convexHull = ms.getConvexHull();
		this.thisLines = new ArrayList<LineSegment2D>(this.convexHull.edges());
	}
	
	public void clip(){
		ArrayList<Point2D> points = new ArrayList<Point2D>();
		while(this.convexHull.vertexNumber()!=2){
			int i = this.convexHull.closestVertexIndex(this.cs.center);
			this.convexHull.removeVertex(i);
		}
		points.addAll(this.convexHull.vertices());
		points.add(this.cs.center);
		this.convexHull = new SimplePolygon2D(points);
		this.thisLines = new ArrayList<LineSegment2D>(this.convexHull.edges());
	}
	
	public boolean containsPoint(Vector point){
		for(LineSegment2D ls : this.thisLines){
			Line2D line = new Line2D(ls.firstPoint(),ls.lastPoint());
			if(line.contains(point)){
				return false;
			}
		}
		return this.convexHull.contains(point);
	}
	
	public void drawTrapezoid(Graphics2D g){
	     this.convexHull.draw(g);
	}
	
	public ArrayList<Point2D> intersection(Ray2D ray){
		ArrayList<Point2D> points = new ArrayList<Point2D>();
		Point2D point = null;
		for(LineSegment2D line : this.thisLines){
			if((point = line.intersection(ray))!=null && !points.contains(point))  points.add(point);
		}
		//System.out.println(points);
		return points;
	}
	
	public boolean intersects(Ray2D ray){
		return this.intersection(ray).size() > 0 ? true : false;
	}
}
