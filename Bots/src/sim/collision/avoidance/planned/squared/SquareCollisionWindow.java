package sim.collision.avoidance.planned.squared;

import java.awt.Color;

import javax.swing.Box;
import javax.swing.ButtonGroup;
import javax.swing.JPanel;
import javax.swing.JRadioButton;

import sim.graphics.Window;
import sim.graphics.World;
import sim.structures.Vector;

@SuppressWarnings("serial")
public class SquareCollisionWindow extends Window{
	
	int numB=10;
	int minSight=5,maxSight=200,numSight=100;
	int bMaxAgentNumber=10,bMinAgentNumber=1,bAgentNumber=4;
	int x=800,y=800;
	
	JPanel panel;
	JRadioButton _16,_36,_64,_100,_144,_196;
	String file="c:/Incoming/Projects/combined_workspace/BotSimulator/output.csv";
	public World createWorld() {
		World world = new World(x,y);
		radioButtonChange();
		Planned p = new Planned(file);
		p.bulkRead();
		for(int i=0;i<p.getAgents();i++){
			Color c= Color.GREEN;
			if(i%10==0) c= Color.BLUE; else if(i%10==1) c= Color.YELLOW; else if(i%10==2) c= Color.MAGENTA; 
			else if(i%10==3) c= Color.RED; else if(i%10==4) c= Color.CYAN; else if(i%10==5) c= Color.GREEN;
			else if(i%10==6) c= Color.GRAY; else if(i%10==7) c= Color.ORANGE; else if(i%10==8) c= Color.PINK;
			else if(i%10==9) c= Color.LIGHT_GRAY;
			PlannedBot b = new PlannedBot(10,c,i,p);
			world.addObject(b,300,340,400,500);
			b.setL(new Vector(p.getMap().get(i).get(0)));
			p.getMap().get(i).remove(0);
		}
		return world;
	}

	@Override
	public JPanel getSettings() {
		panel = new JPanel();
		Box settingsBox = Box.createVerticalBox();
		settingsBox.setAlignmentX(CENTER_ALIGNMENT);
		settingsBox.setAlignmentY(CENTER_ALIGNMENT);
		
		_16 = new JRadioButton("16 Bots");
		_16.setSelected(true);
		_36 = new JRadioButton("36 Bots");
		_64 = new JRadioButton("64 Bots");
		_100 = new JRadioButton("100 Bots");
		_144 = new JRadioButton("144 Bots");
		_196 = new JRadioButton("196 Bots");
		ButtonGroup bg = new ButtonGroup();
		
		bg.add(_16);
		bg.add(_36);
		bg.add(_64);
		bg.add(_100);
		bg.add(_144);
		bg.add(_196);
		
		settingsBox.add(_16);
		settingsBox.add(_36);
		settingsBox.add(_64);
		settingsBox.add(_100);
		settingsBox.add(_144);
		settingsBox.add(_196);
		panel.add(settingsBox);
		return panel;
	}
	
	public void radioButtonChange(){
		if(_16!=null){
			if(_16.isSelected()){ file ="c:/Incoming/Projects/combined_workspace/BotSimulator/output16.csv"; }
			else if(_36.isSelected()) {file ="c:/Incoming/Projects/combined_workspace/BotSimulator/output36.csv"; }
			else if(_64.isSelected()) {file ="c:/Incoming/Projects/combined_workspace/BotSimulator/output64.csv"; }
			else if(_100.isSelected()) {file ="c:/Incoming/Projects/combined_workspace/BotSimulator/output100.csv"; }
			else if(_144.isSelected()) {file ="c:/Incoming/Projects/combined_workspace/BotSimulator/output144.csv"; }
			else if(_196.isSelected()) {file ="c:/Incoming/Projects/combined_workspace/BotSimulator/output196.csv"; }
		}
	}
}
