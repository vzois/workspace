package sim.graphics;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.CyclicBarrier;

import javax.swing.JTextField;

import sim.elements.Bot;
import sim.elements.Thing;
import sim.exec.BalancedThread;
import sim.exec.BotThread;
import sim.structures.Message;
import sim.structures.Vector;
import sim.velocity.obstacles.VelocityObstacle;

import javax.media.opengl.awt.GLJPanel;
import net.sf.javaml.core.kdtree.KDTree;

@SuppressWarnings("serial")
public class World extends GLJPanel{
	protected int limitX;
	protected int limitY;
	public int time;
	protected int population=0;
	
	Color dynamicObstacleColor;
	double dynamicObstacleSize;
	public Random seed = new Random(System.currentTimeMillis());
	
	BotThread []bt;
	BalancedThread []bl;
	
	protected CyclicBarrier cb;
	protected ArrayList<Bot> bots;
 	protected ArrayList<Object> objects;
	protected ArrayList<Message> broadcast;
	protected KDTree kdtree;
	protected JTextField tf;
	protected int speed;
	static ArrayList<VelocityObstacle> vosA=null;
	public static boolean drawVelocityObstacle = false;

	public World(int x,int y){
		super();
		this.limitX=x;
		this.limitY=y;
		this.time = 0;
		this.setAlignmentX(CENTER_ALIGNMENT);
		this.setAlignmentY(CENTER_ALIGNMENT);
		this.setPreferredSize(new Dimension(this.limitX,this.limitY));
		this.setMaximumSize(new Dimension(this.limitX,this.limitY));
		this.setMinimumSize(new Dimension(this.limitX,this.limitY));
		this.objects = new ArrayList<Object>();
		this.bots = new ArrayList<Bot>();
		this.setIgnoreRepaint(true);
		this.kdtree = new KDTree(2);
		World.vosA = new ArrayList<VelocityObstacle>();
	}

	public void enableDrawing(Color color,double size){
		this.dynamicObstacleColor= color;
		this.dynamicObstacleSize = size;
		this.addMouseListener(new MouseListener(){
			@Override
			public void mouseClicked(MouseEvent e) { addDynamicObstacle(e); }
			@Override
			public void mousePressed(MouseEvent e) { }
			@Override
			public void mouseReleased(MouseEvent e) { }
			@Override
			public void mouseEntered(MouseEvent e) { }
			@Override
			public void mouseExited(MouseEvent e) { }
		});
	}
	
	public void setDim(int x, int y){
		this.limitX = x;
		this.limitY = y;
		this.setPreferredSize(new Dimension(this.limitX,this.limitY));
		this.setMaximumSize(new Dimension(this.limitX,this.limitY));
		this.setMinimumSize(new Dimension(this.limitX,this.limitY));
		this.setSize(new Dimension(this.limitX,this.limitY));
		this.repaint();
	}
	
	private void addDynamicObstacle(MouseEvent e){
		Thing t = new Thing(this.dynamicObstacleSize,this.dynamicObstacleColor);
		t.setL(e.getX(), e.getY());
		this.objects.add(t);
		repaint();
	}
	
	public void setSpeed(int speed){
		this.speed = speed;
	}
	
	public int getSpeed(){
		return this.speed;
	}
	
	public void setTextField(JTextField tf){
		this.tf = tf;
	}
	
	public JTextField getTextField(){
		return tf;
	}
	
	public int getLX(){
		return this.limitX;
	}
	
	public int getLY(){
		return this.limitY;
	}
	
	public void setWorld(World world){
		this.objects=world.objects;
		this.bots = world.bots;
		this.population=world.population;
		this.time=0;
		this.dynamicObstacleColor = world.dynamicObstacleColor;
		this.dynamicObstacleSize = world.dynamicObstacleSize;
	}
	
	public void paintObjects(Graphics2D g){
		for(Object o : objects){
			Thing t = (Thing)o;
			g.setColor(((Thing)t).getC());
			g.fill(t.getShape());
			if(o instanceof Bot){
				Bot b = (Bot)o;
				if(Window.draw_velocity && !b.getV().contains(new Vector(0,0))){
					g.setColor(Color.WHITE);
					g.draw(b.getVVShape());
				}
			}
		}
		draw(g);
	}
	
	public void paint(Graphics g){
		super.paint(g);
		render((Graphics2D) g);	
	}
	
	public void render(Graphics2D g)
	{
		super.paint(g);
		paintObjects(g);
	}
	
	public void addObject(Object o){//ADD ELEMENTS RANDOMLY TO THE WORLD
		addObject(o,0,this.limitX,0,this.limitY);
	}
	
	public void addObject(Object o,int minX,int maxX,int minY,int maxY){//ADD OBJECT INTO WORLD//
		if(o instanceof Bot){// IF OBJECT == BOT GIVE IT ID AND SEED FOR RANDOM NUMBERS
			((Bot) o).id = population;
			((Bot) o).seed = new Random(seed.nextLong()*System.currentTimeMillis());
			this.bots.add(((Bot) o));
			population++;
		}
		this.placeObject(o, maxX, minX, maxY, minY);
		this.objects.add(o);
	}
	
	public void placeObject(Object o,int maxX,int minX,int maxY,int minY){// PLACE OBJECTS AND ELIMINATE COLLISIONS//
		boolean conflict =true;
		while(conflict){
			conflict=false;
			((Thing) o).setRL(maxX,minX,maxY,minY, seed);
			for(Object oj : objects){
				conflict = this.collision((Thing)oj, (Thing)o);
				if(conflict) break;
			}
		}
	}
	
	public void initKDTree(){
		this.kdtree = new KDTree(2);
		for(Object o : objects){ insertObject((Thing)o); }
	}
	
	public void insertObject(Thing value){
		double []key = new double[2];
		key[0] = value.getL().getX();
		key[1] = value.getL().getY();
		this.kdtree.insert(key, value);
	}
	
	public void collision(Vector point){
		double []key = new double[2];
		key[0] = point.getX();
		key[1] = point.getY();
		if(this.kdtree==null) System.out.println("tree null");
		Object[] o=this.kdtree.nearest(key,2);
		if(o.length==2){
			Thing other = (Thing)o[1];
			if(other.getCenter().distance(point)<other.getS()){ System.out.println("COLLISION"); }
		}
	}
	
	public Object[] getObjects(Bot value){
		double []lKey = new double[2];
		double []hKey = new double[2];
		lKey[0] = value.getL().getX() - value.getR() - value.getS();
		lKey[1] = value.getL().getY() - value.getR() - value.getS();
		hKey[0] = value.getL().getX() + value.getR() + value.getS();
		hKey[1] = value.getL().getY() + value.getR() + value.getS();
		Object[] o = this.kdtree.range(lKey, hKey);
		return o;
	}
	
	public void simTime3(){
		cb = new CyclicBarrier(population);
		bt = new BotThread[population];
		
		for(int i=0;i<population;i++){
			bt[i] = new BotThread(this,bots.get(i),cb);
			bt[i].setID(i);
			bt[i].start();
		}
	}
	
	public void freezeTime3(){//STOP TIME
		for(int i=0;i<bt.length;i++){
			bt[i].stopT();
		}
	}
	
	public void simTime(){
		int maxNumberOfBotsPerThread = 1000;
		int numberOfBalancedThreads = 0;
		
		if(maxNumberOfBotsPerThread>=population){
			numberOfBalancedThreads = population;
		}else{
			numberOfBalancedThreads = (int)Math.ceil(population/maxNumberOfBotsPerThread);
		}
		
		cb = new CyclicBarrier(numberOfBalancedThreads);
		bl = new BalancedThread[numberOfBalancedThreads];
		for(int i=0;i<bl.length;i++){
			bl[i] = new BalancedThread(this,cb);
			for(int j=i*maxNumberOfBotsPerThread;j<population;j++){
				bl[i].addBot(this.bots.get(j));
			}
			bl[i].setID(i);
			bl[i].start();
		}
	}
	
	public void freezeTime(){//STOP TIME
		for(int i=0;i<bl.length;i++){
			bl[i].stopT();
		}
	}
	
	public boolean collision(Thing a, Thing b){
		double d= a.getL().distance(b.getL());
		if(d>(double)(b.getS() + a.getS())) return false;
		else return true;
	}
	
	public synchronized static void velocity_obstacle(VelocityObstacle vo){
		if(World.vosA!=null){ World.vosA.add(vo); }
	}
	
	public void draw(Graphics2D g){
		
		if((World.vosA!=null)){
			ArrayList<VelocityObstacle> tmp = new ArrayList<VelocityObstacle>(World.vosA);
			for(VelocityObstacle vo : tmp){
				g.setColor(Color.GREEN);
				vo.getTrapezoid().drawTrapezoid(g);
				//g.setColor(Color.ORANGE);
				//vo.getSector().draw(g);
			}
			World.vosA = new ArrayList<VelocityObstacle>();
		}
	}
}
