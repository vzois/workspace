package sim.exec;

import java.util.ArrayList;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

import sim.elements.Bot;
import sim.graphics.World;

public class BotThread extends Thread {
	boolean run=true;
	World world;
	ArrayList<Bot> bots;
	Bot bot;
	CyclicBarrier cb;
	int id;
	
	public BotThread(World world){
		this.bots = new ArrayList<Bot>();
		this.world = world;
		run=true;
	}
	
	public BotThread(World world,Bot bot,CyclicBarrier cb){
		this.world=world;
		this.cb = cb;
		this.bot = bot;
	}
	
	public void addBot(Bot bot){
		this.bots.add(bot);
	}
	
	public void setID(int id){
		this.id = id;
	}
	
	public void run(){
		while(run){
			try {
				Thread.sleep(world.getSpeed());
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
			if(id==0){//THREAD RESPONSIBLE FOR REPAINTING
				world.repaint();
				world.time++;
				world.getTextField().setText(world.time+"");
				world.initKDTree();
			}
			
			world.insertObject(bot);
			synchronize();
			//PERMIT BOT TO ACT//
			bot.setNeighbors(world.getObjects(bot));
			bot.act();
			synchronize();
			bot.move(world.getLX(),world.getLY());//MOVE AROUND THE WORLD
			synchronize();
		}
	}
	
	public void synchronize(){
		try {
			cb.await();
		} catch (InterruptedException | BrokenBarrierException e) {
			e.printStackTrace();
		}
	}
	
	public void stopT(){
		this.run = false;
	}
}
