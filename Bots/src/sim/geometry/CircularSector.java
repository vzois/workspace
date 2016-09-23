package sim.geometry;

import java.awt.Graphics2D;

import math.geom2d.Point2D;
import math.geom2d.line.LineSegment2D;
import math.geom2d.line.Ray2D;
import math.geom2d.polygon.SimplePolygon2D;
import sim.structures.Vector;

public class CircularSector {
	protected Vector center;
	protected Vector leftPoint;
	protected Vector rightPoint;
	protected Vector leftVector;
	protected Vector rightVector;
	protected double radius;
	
	protected SimplePolygon2D sp;
	
	public CircularSector(){
	}
	
	public CircularSector(Vector center,Vector leftPoint, Vector rightPoint,double radius){
		this.center = new Vector(center.getX(),center.getY());
		this.leftPoint = new Vector(leftPoint.getX(),leftPoint.getY());
		this.rightPoint = new Vector(rightPoint.getX(),rightPoint.getY());
		this.radius = radius;
	}
	
	public void center(Vector apex){
		this.center.add(apex);
	}
	
	public void leftPoint(Vector apex){
		this.leftPoint.add(apex);
		this.leftVector = new Vector(this.center,this.leftPoint);
	}
	
	public void rightPoint(Vector apex){
		this.rightPoint.add(apex);
		this.rightVector = new Vector(this.center,this.rightPoint);
	}
	
	public void setCenter(Vector center){
		this.center = new Vector(center.getX(),center.getY());
	}
	
	public void setTangentPoints(Vector leftPoint, Vector rightPoint){
		boolean left = (leftPoint.getY() - this.center.getY())/(leftPoint.getX() - this.center.getX()) 
				> (rightPoint.getY() - this.center.getY())/(rightPoint.getX() - this.center.getX());
				
		if(left){
			this.leftPoint = new Vector(leftPoint.getX(),leftPoint.getY());
			this.leftVector = new Vector(this.center,this.leftPoint);
			this.rightPoint = new Vector(rightPoint.getX(),rightPoint.getY());
			this.rightVector = new Vector(this.center,this.rightPoint);
		}else{
			this.leftPoint = new Vector(rightPoint.getX(),rightPoint.getY());
			this.rightPoint = new Vector(leftPoint.getX(),leftPoint.getY());
			this.rightVector = new Vector(this.center,this.leftPoint);
			this.leftVector = new Vector(this.center,this.rightPoint);
		}
	}
	
	public void setRadius(double radius){
		this.radius = radius;
	}
	
	public Vector getCenter(){
		return this.center;
	}
	
	public double getRadius(){
		return this.radius;
	}
	
	public Vector getLeftPoint(){
		return this.leftPoint;
	}
	
	public Vector getRightPoint(){
		return this.rightPoint;
	}
	
	public boolean containsPoint(Vector point){		
		return this.getPolygon().contains(point);
	}
	
	public void drawCircularSector(Graphics2D g){
		this.getPolygon().draw(g);
	}
	
	public SimplePolygon2D getPolygon(){
		SimplePolygon2D sp = new SimplePolygon2D();
		sp.addVertex(leftPoint);
		sp.addVertex(rightPoint);
		sp.addVertex(center);
		
		return sp;
	}
	
	public void createMinowskiSum(double size){
		SimplePolygon2D A = this.getPolygon();
		SimplePolygon2D B = new SimplePolygon2D();
		
		B.addVertex(new Point2D(size,size));
		B.addVertex(new Point2D(-size,size));
		B.addVertex(new Point2D(size,-size));
		B.addVertex(new Point2D(-size,-size));
		MinkowskiSum ms = new MinkowskiSum(A,B);
		ms.sum();
		this.sp = ms.getConvexHull();
		
		
		while(this.sp.vertexNumber()!=2){
			int i=this.sp.closestVertexIndex(this.center);
			this.sp.removeVertex(i);
		}
		
		SimplePolygon2D local = new SimplePolygon2D(this.sp.vertices());
		local.addVertex(this.center);
		this.sp = local;
	}
	
	public double intersection(Ray2D ray){
		Point2D point = null;
		double minT = Double.MAX_VALUE,tmp;
		for(LineSegment2D l : this.sp.edges()){
			point = l.intersection(ray);
			if(point !=null){
				tmp = ray.positionOnLine(point);
				if(minT>tmp) minT = tmp;
			}
		}		
		return minT;
	}
	
	public boolean intersects(Ray2D ray){
		Point2D point = null;
		for(LineSegment2D l : this.sp.edges()){
			point = l.intersection(ray);
			if(point !=null){ return true; }
		}
		return false;
	}
	
	
	public void draw(Graphics2D g){
		this.sp.draw(g);
	}
	
	
}
