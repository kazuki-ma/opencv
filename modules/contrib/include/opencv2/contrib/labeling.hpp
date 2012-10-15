/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#if __cplusplus && ! defined(__OPENCV_LABELING_HPP__)
#define __OPENCV_LABELING_HPP__ 1


#include <opencv2/core/core.hpp>
#include <vector>
#include <cstdint>


namespace cv{
class Labeling{
	typedef int      image_size_t;   ///< Image Rows and Cols (according to OpenCV, it is int)
	typedef size_t   image_area_t;   ///< Max Area Num
	typedef size_t   image_label_t;  ///< Original Label Type
	
	class LineElement{
		public:
			typedef std::vector<LineElement>::iterator LineElementIterator;

			LineElement()
			: area(0){
			}

			image_size_t size() const {return (col_max - col_min + 1);}

			inline bool isRoot() const{ return (&*this == &*parent);}

			inline LineElementIterator findRoot() {
				if (this->isRoot()) {
					return parent;
				} else {
					return parent = parent->findRoot();
				}
			}

			inline void getRoot() const{
				return (parent == parent->parent) ? parent : parent->getRoot();
			}

			inline void setRoot(const LineElementIterator& new_root) {
				if (! this->isRoot()) {
					parent->setRoot(new_root);
				}
				parent = new_root;
			}

			inline void setParent(const LineElementIterator& new_parent) {
				parent = new_parent;
			}

			inline LineElementIterator getParent() const {
				return parent;
			}

		private:
			LineElementIterator parent;
		public:
			union{image_area_t  area, label;};
			image_size_t  row;     ///< line_element row   position
			image_size_t  col_min; ///< line_element left  position
			image_size_t  col_max; ///< line_element right position
		};


		class RegionInfo {
		public:
			typedef cv::Rect_<image_size_t> Rect;

			RegionInfo()
			{
			}

			RegionInfo(const LineElement& element)
				: area(0)
			{
				border.col_min = element.col_min;
				border.col_max = element.col_max;
				border.row_min = border.row_max = element.row;
				square_accumurate.x = square_accumurate.y = 0;
			}

			size_t getLabel() const {return label;};
			void   setIndex(const image_label_t new_label){ label = new_label; };

			void operator += (const LineElement& element) {
				area += element.size();
				square_accumurate.x = element.size() * (element.col_min + element.col_max) / 1;
				square_accumurate.y = element.size() * element.row;

				if      (element.row < border.row_min) {
					border.row_min = element.row;
				}
				else if (border.row_max < element.row) {
					border.row_max = element.row;
				}

				if (element.col_min < border.col_min) {
					border.col_min = element.col_min;
				}

				if (border.col_max < element.col_max) {
					border.col_max = element.col_max;
				}
			}

		private:
			image_area_t  area;
			image_label_t label;
			struct{image_size_t col_min, col_max, row_min, row_max;} border;
			typedef struct{double x, y;}                        xy;

			union{
				xy center_of_gravity;
				xy square_accumurate;
			};
		};

	public:

		typedef cv::Vector<RegionInfo>   RegionInfoVec;
		typedef std::vector<LineElement> LineElementVec;
		typedef LineElementVec::iterator LineElementVecIterator;

		enum LIMIT_TYPE{
			LIMIT_AREA_SIZE,
			LIMIT_NOF_AREA
		};
		
		Labeling(image_label_t element_estimate = 1024)
		: line_elements(element_estimate){
		}

		virtual ~Labeling(void){};

		template <typename SRC_TYPE, typename DST_TYPE>
		Labeling exec(const cv::Mat_<SRC_TYPE> &src, cv::Mat_<DST_TYPE> &dst) {
			assert(src.dims == 2 && src.rows > 0 && src.cols > 0);
			dst.create(src.size());
			line_elements.resize(0);
			regions = RegionInfoVec();

			p000_make_line_elements_from_source(src);
			p010_make_tree_structureof_line_elements();
			p020_totalize_tree_cost();
			p030_fixing_label_and_enforce_limit();
			p040_make_labeled_image(dst);

			return *this;
		}

		const RegionInfoVec getRegions() const{
			return regions;
		}

	private:
		LineElementVec line_elements;
		RegionInfoVec  regions;

		/// Functions

		/*!
			Analayze image and extract consective non-zero element in a row
			and push the element into vector.
		*/
		template <typename SRC_TYPE>
		void p000_make_line_elements_from_source(const cv::Mat_<SRC_TYPE> &src) {
			assert(src.type() == cv::DataType<SRC_TYPE>::type);

			const SRC_TYPE *src_ptr;
			const int r_max = src.rows;
			const int c_max = src.cols;

			LineElement current;

			for (image_size_t r = 0; r < r_max; ++r) {
				src_ptr = src.ptr<SRC_TYPE>(r);

				for (image_size_t c = 0; c < c_max; ++c) {
					/// find first non-zero

					while(0 == src_ptr[c] && (c < c_max)) {
						c++;
					}

					/// Line Element Beggining Mark
					current.row     = r;
					current.col_min = c;

					/// find last zero
					while (c < c_max) {
						if (src_ptr[c]) c++; if (src_ptr[c]) c++;
						if (src_ptr[c]) c++; if (src_ptr[c]) c++; else break;
					}

					current.col_max = c - 1;
					line_elements.push_back(current);
				}
			}
    }

		/*!
		Determin which element is parent and its child and making edge by set parent of child.
		*/
		inline void make_edge(LineElementVecIterator &fore, LineElementVecIterator &back) {
			//fore->findRoot();
			//back->findRoot();

			bool fore_is_root = fore->isRoot();
			bool back_is_root = back->isRoot();

			if       (fore_is_root && ! back_is_root) {
				// fore だけが孤立エレメントの時，
				// fore の link を back の大元に書き換えて終わり
				fore->setParent(back->findRoot());
			}
			else if (fore_is_root && back_is_root) {
				// どちらも孤立エレメントならば，
				// 上の方のエレメントが上のコストを吸収する
				fore->setParent(back);
			}
			else if (back_is_root) {
				// back だけが孤立エレメントの場合，
				// （foreにとって二つ目以降の結合エレメントの場合）
				// back のコストを fore で吸収する
				back->setParent(fore->findRoot());
			}
			else {
				// どちらも孤立エレメントで無い場合
				// back 側の親で吸収する．
				if  (fore->findRoot() == back->findRoot()) {
					return;
				}

				fore->setRoot(back->findRoot());
			}
		}


		/*!
			Make tree structure from extracted line elements.
		*/
		void p010_make_tree_structureof_line_elements() {
			for (
				LineElementVecIterator it = line_elements.begin();
				it != line_elements.end();
			++it)
			{
				it->setParent(it);
			}

			LineElement centinel;
			centinel.row = std::numeric_limits<image_size_t>::max();
			line_elements.push_back(centinel);

			LineElementVecIterator it_f         = line_elements.begin() + 1;
			LineElementVecIterator it_b         = line_elements.begin();
			LineElementVecIterator it_last_f    = line_elements.begin();
			const LineElementVecIterator it_end = line_elements.end();

			while (it_f != it_end) {
				// Backword が Forward の3行以上手前ならば Backword を進める．
				if (2 < it_f->row - it_b->row)
					it_b = it_f;
				else
					it_b = it_last_f;

				// 結果追いついてしまったら it_f を進めてループに再入する
				if (it_b->row == it_f->row) {
					while(it_f != it_end && it_b->row == it_f->row) {
						it_f++;
					}
					continue;
				}

				it_last_f = it_f;


				/// Tick Tack 方式で line_element の被りを見ていく
				while (1 == (it_f->row - it_b->row)) {
					if      (it_f->col_max < it_b->col_min) {
						it_f++; ///< forward 要素の方が左にあって重複していない場合
					}
					else if (it_b->col_max < it_f->col_min) {
						it_b++; ///< forward 要素の方が右にあって重複していない場合
					}
					else { /// 重複した場合
						make_edge(it_f, it_b);

						if (it_b->col_max < it_f->col_max) {
							it_b++;
						} else {
							it_f++;
						}
					}
				}
			}
			line_elements.pop_back();
		}


		/*!
			Set total cost of tree on the cost of the root element.
		*/
		void p020_totalize_tree_cost() {
			for (
				LineElementVecIterator it = line_elements.begin()
				; it != line_elements.end()
				; ++it)
			{
				it->findRoot()->area += it->size();
			}
		}

		void p031_set_label(std::vector<LineElementVecIterator> &line_elements,
			const int label_limit_type, const image_label_t label_limit) 
		{
			size_t   idx = 0;
			switch (label_limit_type) {
			case LIMIT_AREA_SIZE:

				for (; idx < line_elements.size() && label_limit <= line_elements[idx]->area; ++idx) {
					regions.push_back(*line_elements[idx]);
					line_elements[idx]->label = idx + 1;
				}
				for (; idx < line_elements.size(); ++idx) {
					line_elements[idx]->label = 0;
				}

				break;

			case LIMIT_NOF_AREA:
				{
					const auto regions_limit = std::min<image_label_t>(line_elements.size(), label_limit);

					for (; idx < regions_limit; ++idx) {
						regions.push_back(*line_elements[idx]);
						line_elements[idx]->label = idx + 1;
					}
					for (; idx < line_elements.size(); ++idx) {
						line_elements[idx]->label = 0;
					}
				}
				break;
			default:
				assert(true);
			}

		}

		/*!
			Sort a poitner of line elements by its area.
			and enforce limit.
			Set are = 0 if the element is in a the limit border.
			And create regions from rooot elements.
		*/
		void p030_fixing_label_and_enforce_limit(int label_limit_type = LIMIT_AREA_SIZE, int label_limit = 1) {
			std::vector<LineElementVecIterator> sort_temp;
			sort_temp.reserve(line_elements.size());

			for (LineElementVecIterator it = line_elements.begin()
				; it != line_elements.end()
				; ++it )
			{
				if (it->isRoot())
					sort_temp.push_back(it);
			}

			struct{
				bool operator() (const LineElementVecIterator lhs, const LineElementVecIterator rhs) {
					return lhs->area > rhs->area;
				}
			} line_element_compare;
			std::sort(sort_temp.begin(), sort_temp.end(), line_element_compare);

			p031_set_label(sort_temp, label_limit_type, label_limit);

			for (LineElementVec::iterator it = line_elements.begin()
				; it != line_elements.end()
				; ++it )
			{
				const image_label_t label = it->getParent()->label;
				if (label != 0) {
					it->label = label;
					regions[label - 1] += *it;
				}
			}
		}

		template <typename DST_TYPE>
		void p040_make_labeled_image(cv::Mat_<DST_TYPE> &dst) {
			dst.setTo(0);

			for (
				LineElementVecIterator it = line_elements.begin();
				it != line_elements.end();
			it++)
			{
				const DST_TYPE label = it->label;
				if (0 != label) {
					DST_TYPE * const ptr = dst.ptr<DST_TYPE>(it->row);

					for (image_size_t c = it->col_min; c <= it->col_max; ++c) {
						ptr[c] = label;
					}
				}
			}
		};
	}; // end class Labeling
}

#endif //#ifndef __OPENCV_LABELING_HPP__
